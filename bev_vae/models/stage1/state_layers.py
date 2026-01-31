from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from bev_vae.models.layers import DePatchSceneEmbed, PatchSceneEmbed
from bev_vae.models.rope import RoFormerPE3D


class StateEncoder(nn.Module):
    def __init__(
        self, 
        scene_size: List[int], 
        patch_size: int,
        in_ch: int,
        ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.grid_size = (scene_size[0], scene_size[1]//patch_size, scene_size[2]//patch_size)
        self.to_state = PatchSceneEmbed(self.grid_size[0], patch_size, in_ch, ch, out_ch)
        self.transformer = RoFormerPE3D(self.grid_size, out_ch * self.grid_size[0], block_num, head_num, dropout, bias, flash_attn, ckpt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_state(x) # b e d H W -> b d h w e  
        x = rearrange(x, "b d h w e -> b (h w) (d e)")
        return self.transformer(x)
    
class StateDecoder(nn.Module):
    def __init__(
        self, 
        scene_size: List[int], 
        patch_size: int,
        in_ch: int,
        ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.grid_size = (scene_size[0], scene_size[1]//patch_size, scene_size[2]//patch_size)
        self.transformer = RoFormerPE3D(self.grid_size, in_ch*self.grid_size[0], block_num, head_num, dropout, bias, flash_attn, ckpt)
        self.to_scene = DePatchSceneEmbed(self.grid_size[0], patch_size, in_ch, ch, out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = rearrange(x, "b (h w) (d e) -> b d h w e", d=self.grid_size[0], h=self.grid_size[1])
        return self.to_scene(x)