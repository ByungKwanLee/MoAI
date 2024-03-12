# # Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InternLM2 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging)

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

from .build_mlp import PLoRA
from .configuration_internlm2 import InternLM2Config
logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.tensor(torch.finfo(dtype).min, device=device),
                      device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ],
                         dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len,
    src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min)


class InternLM2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """InternLM2RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM2RotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            **(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)


class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len /
                                 self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(
                                    self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                **(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    if q.size(2) == 1:
        q_embed = (q * cos[:, :, -1:, :]) + (
            rotate_half(q) * sin[:, :, -1:, :])
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    if k.size(2) == 1:
        k_embed = (k * cos[:, :, -1:, :]) + (
            rotate_half(k) * sin[:, :, -1:, :])
    else:
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class InternLM2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = PLoRA(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self.w3 = PLoRA(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self.w2 = PLoRA(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, im_mask):
        down_proj = self.w2(
            self.act_fn(self.w1(x, im_mask)) * self.w3(x, im_mask), im_mask=im_mask)

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class InternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: InternLM2Config, moai:bool):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.is_add_MoAI_mixer = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).')

        self.wqkv = PLoRA(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)

        self.wo = PLoRA(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self._init_rope()

        if moai:
            # MoAI-CA/SA
            from .expert_module import MultiHeadLoRAAttention
            self.moai_CA_img_aux = MultiHeadLoRAAttention()
            self.moai_CA_img_lang = MultiHeadLoRAAttention()
            self.moai_SA_img = MultiHeadLoRAAttention()
            self.moai_CA_lang_aux = MultiHeadLoRAAttention()
            self.moai_CA_lang_img = MultiHeadLoRAAttention()
            self.moai_SA_lang = MultiHeadLoRAAttention()
            
            # MoAI-Gating Network
            self.moai_GA_img = nn.Linear(4096, 3)
            self.moai_GA_lang = nn.Linear(4096, 3)
            
            # flag add MoAI mixer
            self.is_add_MoAI_mixer = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor)
            else:
                raise ValueError(
                    "Currently we only support rotary embedding's type being 'dynamic'."
                )
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        im_mask: Optional[Tuple[torch.Tensor]] = None,
        lang_mask: Optional[Tuple[torch.Tensor]] = None,
        mode: str = 'moai_eval.yaml',
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`')

        bsz, q_len, _ = hidden_states.size()

        # is_add_MoAI_mixer
        if self.is_add_MoAI_mixer:
            if mode == 'moai_step1.yaml':
                # Shallow Copy
                h_im = hidden_states[im_mask]
                h_lang = hidden_states[lang_mask]

                # rolling dice
                rolling_dice = torch.randint(low=0, high=3, size=(1,))
                                            
                # MoAI-IMG
                if rolling_dice==0:
                    mix_img = self.moai_CA_img_aux(h_im, aux_embeds.squeeze(dim=0), aux_embeds.squeeze(dim=0))
                elif rolling_dice==1:
                    mix_img = self.moai_CA_img_lang(h_im, h_lang, h_lang)
                elif rolling_dice==2:
                    mix_img = self.moai_SA_img(h_im, h_im, h_im)
                
                # rolling dice
                rolling_dice = torch.randint(low=0, high=3, size=(1,))

                # MoAI-Lang
                if rolling_dice==0:
                    mix_lang = self.moai_CA_lang_aux(h_lang, aux_embeds.squeeze(dim=0), aux_embeds.squeeze(dim=0))
                elif rolling_dice==1:
                    mix_lang = self.moai_CA_lang_img(h_lang, h_im, h_im)
                elif rolling_dice==2:
                    mix_lang = self.moai_SA_lang(h_lang, h_lang, h_lang)

                # Mix
                hidden_states[im_mask] = mix_img
                hidden_states[lang_mask] = mix_lang

            elif mode == 'moai_step2.yaml':
                import torch.nn.functional as F
                
                # Shallow Copy
                h_im = hidden_states[im_mask]
                h_lang = hidden_states[lang_mask]
                
                # Gating Network
                soft_img_weight = F.softmax(self.moai_GA_img(h_im), dim=1, dtype=torch.bfloat16)
                soft_lang_weight = F.softmax(self.moai_GA_lang(h_lang), dim=1, dtype=torch.bfloat16)
                
                # MoAI-IMG
                mix_img =\
                soft_img_weight[..., 0].unsqueeze(dim=1) * self.moai_CA_img_aux(h_im, aux_embeds.squeeze(dim=0), aux_embeds.squeeze(dim=0))\
                + soft_img_weight[..., 1].unsqueeze(dim=1) * self.moai_CA_img_lang(h_im, h_lang, h_lang)\
                + soft_img_weight[..., 2].unsqueeze(dim=1) * self.moai_SA_img(h_im, h_im, h_im)
                
                # MoAI-Lang
                mix_lang =\
                soft_lang_weight[..., 0].unsqueeze(dim=1) * self.moai_CA_lang_aux(h_lang, aux_embeds.squeeze(dim=0), aux_embeds.squeeze(dim=0))\
                + soft_lang_weight[..., 1].unsqueeze(dim=1) * self.moai_CA_lang_img(h_lang, h_im, h_im)\
                + soft_lang_weight[..., 2].unsqueeze(dim=1) * self.moai_SA_lang(h_lang, h_lang, h_lang)
                
                # Mix
                hidden_states[im_mask] = mix_img
                hidden_states[lang_mask] = mix_lang

            elif mode == 'moai_eval.yaml':
                import torch.nn.functional as F

                if sum(im_mask[0])!=0 or sum(lang_mask[0])!=0:
                    
                    for beam_index in range(hidden_states.shape[0]):
                        # Shallow Copy
                        h_im = hidden_states[beam_index][im_mask[beam_index]]
                        h_lang = hidden_states[beam_index][lang_mask[beam_index]]
                        
                        # Gating Network
                        soft_img_weight = F.softmax(self.moai_GA_img(h_im), dim=1, dtype=torch.float16)
                        soft_lang_weight = F.softmax(self.moai_GA_lang(h_lang), dim=1, dtype=torch.float16)
                        
                        # MoAI-IMG
                        mix_img =\
                        soft_img_weight[..., 0].unsqueeze(dim=1) * self.moai_CA_img_aux(h_im, aux_embeds[beam_index], aux_embeds[beam_index])\
                        + soft_img_weight[..., 1].unsqueeze(dim=1) * self.moai_CA_img_lang(h_im, h_lang, h_lang)\
                        + soft_img_weight[..., 2].unsqueeze(dim=1) * self.moai_SA_img(h_im, h_im, h_im)
                        
                        # MoAI-Lang
                        mix_lang =\
                        soft_lang_weight[..., 0].unsqueeze(dim=1) * self.moai_CA_lang_aux(h_lang, aux_embeds[beam_index], aux_embeds[beam_index])\
                        + soft_lang_weight[..., 1].unsqueeze(dim=1) * self.moai_CA_lang_img(h_lang, h_im, h_im)\
                        + soft_lang_weight[..., 2].unsqueeze(dim=1) * self.moai_SA_lang(h_lang, h_lang, h_lang)
                        
                        # Mix
                        hidden_states[beam_index][im_mask[beam_index]] = mix_img
                        hidden_states[beam_index][lang_mask[beam_index]] = mix_lang

            else:
                raise Exception("Not Registered Mode!")
        
        
        qkv_states = self.wqkv(hidden_states, im_mask)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}')

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output, im_mask)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLM2DecoderLayer(nn.Module):

    def __init__(self, config: InternLM2Config, moai:bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = InternLM2Attention(config=config, moai=moai)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aux_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        im_mask: Optional[Tuple[torch.Tensor]] = None,
        lang_mask: Optional[Tuple[torch.Tensor]] = None,
        mode: str = 'moai_eval.yaml',
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`')

        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            aux_embeds=aux_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            im_mask=im_mask,
            lang_mask=lang_mask,
            mode=mode,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, im_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternLM2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class InternLM2Model(InternLM2PreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = 'AutoModel'

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size+1,
                                           config.hidden_size,
                                           self.padding_idx)
        self.layers = nn.ModuleList([
            InternLM2DecoderLayer(config, moai=(ind+1) % 8 == 0)
            for ind in range(config.num_hidden_layers)
        ])
        self.norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:

        aux_embeds = kwargs.get('aux_embeds', None)
        im_mask = kwargs.get('im_mask', None)
        lang_mask = kwargs.get('lang_mask', None)
        mode = kwargs.get('mode', 'moai_eval.yaml')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
            im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                inputs_embeds.device).bool()
            lang_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                inputs_embeds.device).bool()
            aux_embeds = None
            mode = 'moai_eval.yaml'
            assert False
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training and (idx+1)%8==0:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None,
                                      im_mask=im_mask, lang_mask=lang_mask, mode=mode)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    aux_embeds,
                    attention_mask,
                    position_ids,
                    None,
                    use_reentrant=True
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    aux_embeds=aux_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    im_mask=im_mask,
                    lang_mask=lang_mask,
                    mode=mode,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )