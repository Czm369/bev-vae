import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, embed_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-6)


class MLP(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RearrangeContiguous(nn.Module):
    def __init__(self, pattern, **kwargs):
        super().__init__()
        self.rearrange = Rearrange(pattern, **kwargs)

    def forward(self, x):
        return self.rearrange(x).contiguous()


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('b e h w -> b h w e'))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    

class DePatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            RearrangeContiguous('b h w e -> b e h w'),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchSceneEmbed(nn.Module):
    def __init__(self, height: int, patch_size: int, in_ch: int, ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, ch, 1), nn.GELU(),
            RearrangeContiguous('b e d h w -> (b d) e h w'),
            nn.Conv2d(ch, out_ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('(b d) e h w -> b d h w e', d=height))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    

class DePatchSceneEmbed(nn.Module):
    def __init__(self, height: int, patch_size: int, in_ch: int, ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            RearrangeContiguous('b d h w e -> (b d) e h w'),
            nn.ConvTranspose2d(in_ch, ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('(b d) e h w -> b e d h w', d=height),
            nn.GELU(), nn.Conv3d(ch, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  


class MultiScaleDefAttn2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.level_num = level_num
        self.point_num = point_num
        self.dropout = nn.Dropout(dropout)
        self.offset = nn.Linear(embed_dim, head_num * level_num * point_num *  2, bias)
        self.weight = nn.Linear(embed_dim, head_num * level_num * point_num, bias)
        # self.value_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        cam_num = valid.shape[0]
        # query: B (DHW) E | value: V sum(imgHW) B E | point: V B (DHW) xy | valid: V B (DHW)
        # different sample in a batch must be same geom augmented in the same view 
        indexes = [valid_per_view[0].nonzero().squeeze(-1) for valid_per_view in valid]
        query_num = max([len(valid_per_view) for valid_per_view in indexes])
        b, _, _ = query.shape # B (DHW) E
        query_rebatch = query.new_zeros([b, cam_num, query_num, self.embed_dim])
        point_rebatch = point.new_zeros([b, cam_num, query_num, 2])
        for v in range(cam_num):
            query_rebatch[:, v, :len(indexes[v])] = query[:, indexes[v]]
            point_rebatch[:, v, :len(indexes[v])] = point[v, :, indexes[v]]
        _, value_num, _, _ = value.shape # V sum(imgHW) B C 
        output = self.cross_attn(
            query_rebatch.reshape(b * cam_num, query_num, self.embed_dim), 
            value.permute(2, 0, 1, 3).reshape(b * cam_num, value_num, self.embed_dim), 
            point_rebatch.reshape(b * cam_num, query_num, 2), 
            shape).reshape(b, cam_num, query_num, self.embed_dim)
        slots = torch.zeros_like(query) # B (DHW) E
        for v in range(cam_num):
            slots[:, indexes[v]] += output[:, v, :len(indexes[v])]
        count = valid.permute(1, 2, 0).sum(-1).clamp(min=1.0).to(slots.dtype) # B (DHW) sum(V)
        slots = self.proj(slots / count[..., None])
        return self.dropout(slots) 

    def cross_attn(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        # query: (BV) query_num E | value: (BV) sum(imgHW) E | point: (BV) query_num 2 
        b_cam, query_num, _ = query.shape
        _, value_num, _ = value.shape
        e = self.embed_dim // self.head_num
        assert shape.prod(1).sum() == value_num
        # value = self.value_proj(value)
        offset = self.offset(query).reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num, 2)
        weight = self.weight(query).reshape(
            b_cam, query_num, self.head_num, self.level_num * self.point_num).softmax(-1)
        weight = weight.reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num)
        location = point[:, :, None, None, None, :] + offset / shape[None, None, None, :, None, :].flip(-1)
        grid = location * 2 - 1
        value_list = value.split([h * w for h, w in shape], dim=1)
        sampling_value_list = []
        for lvl, (h, w) in enumerate(shape):
            value_lvl = value_list[lvl].transpose(1, 2).reshape(b_cam * self.head_num, e, h, w)
            grid_lvl = grid[:, :, :, lvl].transpose(1, 2).flatten(0, 1) # (b_cam head_num) query_num point_num xy
            sampling_value_list.append(F.grid_sample(value_lvl, grid_lvl, align_corners=False)) # (b_cam head_num) e query_num point_num
        weight = weight.transpose(1, 2).reshape(b_cam * self.head_num, 1, query_num, self.level_num * self.point_num)
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2) * weight
        # (b_cam head_num) e query_num (level_num point_num)
        output = rearrange(output.sum(-1), "(bv h) e q -> bv q (h e)", h=self.head_num)
        return output # (BV) query_num E
    

class MSDABlock2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_v = LayerNorm(embed_dim, bias)
        self.attn = MultiScaleDefAttn2D(embed_dim, head_num, level_num, point_num, dropout, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        query = query + self.attn(self.ln_1(query), self.ln_v(value), point, valid, shape)
        query = query + self.mlp(self.ln_2(query))
        return query
    

class MSDATransformer2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSDABlock2D(embed_dim, head_num, level_num, point_num, dropout, bias) for _ in range(block_num)])
        self.norm = LayerNorm(embed_dim, bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            if self.ckpt and self.training:
                query = checkpoint.checkpoint(block, query, value, point, valid, shape, use_reentrant=False) 
            else:
                query = block(query, value, point, valid, shape)
        return self.norm(query)


class MultiScaleDefAttn3D(nn.Module):
    def __init__(self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.level_num = level_num
        self.point_num = point_num
        self.depth_num = depth_num
        self.dropout = nn.Dropout(dropout)
        self.offset = nn.Linear(embed_dim, head_num * level_num * point_num * 3, bias)
        self.weight = nn.Linear(embed_dim, head_num * level_num * point_num, bias)
        self.depth = nn.Linear(embed_dim, 1, bias)
        # self.value_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor: 
        # query: B V (DHW) E | value: sum(scnDHW) B E | point: V B (DHW) xyz | valid: V B (DHW)
        # different sample in a batch must be same geom augmented in the same view 
        cam_num = valid.shape[0]
        indexes = [valid_per_view[0].nonzero().squeeze(-1) for valid_per_view in valid]
        query_num = max([len(valid_per_view) for valid_per_view in indexes])
        b, _, _, _ = query.shape # B V (DHW) E
        query_rebatch = query.new_zeros([b, cam_num, query_num, self.embed_dim])
        point_rebatch = point.new_zeros([b, cam_num, query_num, 3])
        for v in range(cam_num):
            query_rebatch[:, v, :len(indexes[v])] = query[:, v, indexes[v]]
            point_rebatch[:, v, :len(indexes[v])] = point[v, :, indexes[v]]
        output = self.cross_attn(
            query_rebatch.reshape(b * cam_num, query_num, self.embed_dim),
            value.permute(1, 0, 2),
            point_rebatch.reshape(b * cam_num, query_num, 3),
            shape).reshape(b, cam_num, query_num, self.embed_dim)
        slots = torch.zeros_like(query) # B V (DHW) E
        for v in range(cam_num):
            slots[:, v, indexes[v]] += output[:, v, :len(indexes[v])]
        slots = rearrange(slots, "b v (d q) e -> b v q e d", d = self.depth_num)
        depth = self.depth(query) # B V (DHW) 1
        depth = depth.masked_fill(~valid.permute(1, 0, 2)[..., None], -torch.finfo(depth.dtype).max)
        depth = rearrange(depth, "b v (d q) e -> b v q e d ", d=self.depth_num)
        depth = depth.softmax(-1)
        slots = self.proj((slots * depth).sum(-1)) # B V Q E
        slots = slots[:, :, None].repeat(1, 1, self.depth_num, 1, 1).flatten(2, 3)
        return self.dropout(slots) 

    def cross_attn(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        point: torch.Tensor,
        shape: List[torch.Tensor]) -> torch.Tensor:
        # query: (BV) query_num E | value: B sum(scnHW) E | point: (BV) query_num 3      
        b_cam, query_num, _ = query.shape
        b, value_num, _ = value.shape
        cam_num = b_cam // b
        e = self.embed_dim // self.head_num
        assert shape.prod(1).sum() == value_num
        # value = self.value_proj(value)
        offset = self.offset(query).reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num, 3)
        weight = self.weight(query).reshape(
            b_cam, query_num, self.head_num, self.level_num * self.point_num).softmax(-1)
        weight = weight.reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num)
        location = point[:, :, None, None, None, :] + offset / shape[None, None, None, :, None, :].flip(-1)
        grid = rearrange(location * 2 - 1, "(b v) q ... -> b (v q) ...", b=b)
        value_list = value.split([d * h * w for d, h, w in shape], dim=1)
        sampling_value_list = []
        for lvl, (d, h, w) in enumerate(shape):
            value_lvl = value_list[lvl].transpose(1, 2).reshape(b * self.head_num, e, d, h, w)
            grid_lvl = grid[:, :, :, lvl].transpose(1, 2).flatten(0, 1)[..., None, :] # (b head_num) (cam_num query_num) point_num 1 xyz
            sampling_value_list.append(F.grid_sample(value_lvl, grid_lvl, align_corners=False).squeeze(-1)) # (b head_num) e (cam_num query_num) point_num
        weight = rearrange(weight, "(b v) q ... -> b (v q) ...", b=b)
        weight = weight.transpose(1, 2).reshape(b * self.head_num, 1, cam_num * query_num, self.level_num * self.point_num)
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2) * weight # (b head_num) e (cam_num query_num) (level_num point_num)
        output = rearrange(output.sum(-1), "(b h) e (v q) -> (b v) q (h e)", h=self.head_num, v=cam_num)
        return output # (BV) query_num E
    

class MSDABlock3D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_v = LayerNorm(embed_dim, bias)
        self.attn = MultiScaleDefAttn3D(embed_dim, head_num, level_num, point_num, depth_num, dropout, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        query = query + self.attn(self.ln_1(query), self.ln_v(value), point, valid, shape)
        query = query + self.mlp(self.ln_2(query))
        return query


class MSDATransformer3D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSDABlock3D(embed_dim, head_num, level_num, point_num, depth_num, dropout, bias) for _ in range(block_num)])
        self.depth_num = depth_num
        self.norm = LayerNorm(embed_dim, bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt
        
    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            if self.ckpt and self.training:
                query = checkpoint.checkpoint(block, query, value, point, valid, shape, use_reentrant=False)
            else:
                query = block(query, value, point, valid, shape)
        query = rearrange(query, "b v (d q) e -> b v q e d", d=self.depth_num)
        return self.norm(query.mean(-1))
    