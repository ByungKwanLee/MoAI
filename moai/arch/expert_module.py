import torch
from torch import nn, einsum
import torch.nn.functional as F
import math


from einops import rearrange, repeat
from einops_exts import rearrange_many

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim=4096,
        depth=4,
        dim_head=64,
        heads=4,
        num_latents=64,
        ff_mult=1
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents).squeeze(dim=1)
    
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadLoRAAttention(nn.Module):

    def __init__(self,
                 in_features=4096,
                 head_num=4,
                 bias=True,
                 activation=F.gelu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadLoRAAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Sequential(nn.Linear(in_features, 64, False), nn.Linear(64, in_features, False))
        self.linear_k = nn.Sequential(nn.Linear(in_features, 64, False), nn.Linear(64, in_features, False))
        self.linear_v = nn.Sequential(nn.Linear(in_features, 64, False), nn.Linear(64, in_features, False))
        self.linear_o = nn.Sequential(nn.Linear(in_features, 64, False), nn.Linear(64, in_features, False))

    def forward(self, q, k, v, mask=None):
        _q, _k, _v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            _q = self.activation(_q)
            _k = self.activation(_k)
            _v = self.activation(_v)

        _q = self._reshape_to_batches(_q)
        _k = self._reshape_to_batches(_k)
        _v = self._reshape_to_batches(_v)
        if mask is not None: mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(_q, _k, _v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None: y = self.activation(y)
        return q + y

    def _reshape_to_batches(self, x):
        seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(seq_len, self.head_num, sub_dim)\
                .permute(1, 0, 2)

    def _reshape_from_batches(self, x):
        self.head_num, seq_len, in_feature = x.size()
        return x.permute(1, 0, 2)\
                .reshape(seq_len, -1)