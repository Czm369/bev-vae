from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from bev_vae.models.layers import DePatchEmbed, PatchEmbed
from bev_vae.models.rope import RoFormerPE2D


class ImageEncoder(nn.Module):
    def __init__(
        self, 
        image_size: List[int], 
        patch_size: int,
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.grid_size = (image_size[0]//patch_size, image_size[1]//patch_size)
        self.to_patch = PatchEmbed(patch_size, in_ch, out_ch)
        self.transformer = RoFormerPE2D(self.grid_size, out_ch, block_num, head_num, dropout, bias, flash_attn, ckpt)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch(x) # B H W E
        x = rearrange(x, "b h w e -> b (h w) e")
        x = self.transformer(x)
        x = rearrange(x, "b (h w) e -> b e h w", h=self.grid_size[0])
        return x.contiguous()
    

class ImageDecoder(nn.Module):
    def __init__(
        self, 
        grid_size: List[int],
        patch_size: int,
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.transformer = RoFormerPE2D(self.grid_size, in_ch, block_num, head_num, dropout, bias, flash_attn, ckpt)
        self.to_pixel = DePatchEmbed(patch_size, in_ch, out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b e h w -> b (h w) e")
        x = self.transformer(x.contiguous())
        x = rearrange(x, "b (h w) e -> b h w e", h=self.grid_size[0])
        x = self.to_pixel(x)
        return x
    