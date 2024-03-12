import math
import re

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


def build_vision_tower():
    vision_tower = 'openai/clip-vit-large-patch14-336'
    return CLIPVisionTower(vision_tower)


def build_vision_projector():
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 1024
    hidden_size = 4096

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {'mm_projector_type': 'identity'}


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower):
        super().__init__()

        self.is_loaded = False
        self.is_resize_pos = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = 'patch'
        self.load_model()
        self.resize_pos()

    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def resize_pos(self):
        pos_embed_checkpoint = self.vision_tower.vision_model.embeddings.position_embedding.weight
        pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(0)
        orig_size = 24
        new_size = 35

        if pos_embed_checkpoint.shape[1] == new_size**2 + 1:
            self.is_resize_pos = True
        else:
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1
            new_num = new_size**2 + num_extra_tokens
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(
                                                0, 3, 1, 2).float()
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).half()
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            new_pos_embed = new_pos_embed.squeeze(0)

            self.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
                new_num, 1024)
            self.vision_tower.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(
                new_pos_embed.to(pos_embed_checkpoint.dtype))
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(
                new_num).expand((1, -1))

            self.is_resize_pos = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(
                f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device,
                             dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(
                images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(
            1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2


class PLoRA(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 lora_len=0,
                 **kwargs) -> None:
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_len = lora_len
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.Plora_A = nn.Linear(
            in_features, self.lora_r, bias=False, device=device, dtype=dtype)
        self.Plora_B = nn.Linear(
            self.lora_r, out_features, bias=False, device=device, dtype=dtype)
        
        # LBK
        self.Plora_main = nn.Linear(in_features, out_features, bias, device, dtype)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)


    # LBK
    def forward(self, x, im_mask):
        res = self.Plora_main(x)
        if im_mask is not None:
            if torch.sum(im_mask) > 0:
                part_x = x[im_mask]
                res[im_mask] += self.Plora_B(
                    self.Plora_A(
                        self.lora_dropout(part_x))) * self.lora_scaling
            else:
                part_x = x[:, :1]
                res[:, :1] += self.Plora_B(
                    self.Plora_A(self.lora_dropout(part_x))) * 0
        return res