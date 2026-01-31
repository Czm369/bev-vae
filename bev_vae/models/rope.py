import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def rope(pos: Tensor, inv_freq: Tensor) -> Tensor:
    angle = torch.einsum("...n,d->...nd", pos, inv_freq)
    emb = torch.stack([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], dim=-1)
    emb = rearrange(emb, "n d (i j) -> n d i j", i=2, j=2)
    return emb.float()


def apply_rope(q: Tensor, k: Tensor, pe: Tensor) -> Tensor:
    xq = q.float().reshape(*q.shape[:-1], -1, 1, 2)
    xk = k.float().reshape(*k.shape[:-1], -1, 1, 2) # b nh T hs -> b nh T 32 1 2
    freqs_cis = pe[None].repeat(q.shape[0], 1, 1, 1, 1, 1)       # b 1 T 32 2 2
    xq_out = freqs_cis[..., 0] * xq[..., 0] + freqs_cis[..., 1] * xq[..., 1]
    xk_out = freqs_cis[..., 0] * xk[..., 0] + freqs_cis[..., 1] * xk[..., 1]
    return xq_out.reshape(*q.shape).type_as(q), xk_out.reshape(*k.shape).type_as(k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class RoSelfAttn(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool) -> None:
        super().__init__()

        assert embed_dim % head_num == 0

        head_dim = embed_dim // head_num
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.norm = QKNorm(head_dim)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.dropout = dropout
        self.flash_attn = flash_attn

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.head_num, C // self.head_num).transpose(1, 2).contiguous() # (B, nh, T, hs)
        k = k.view(B, T, self.head_num, C // self.head_num).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(B, T, self.head_num, C // self.head_num).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)

        if self.flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, embed_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-6)


class RoBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.attn = RoSelfAttn(embed_dim, head_num, dropout, bias, flash_attn)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x), pe)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class RoFormerPE2D(nn.Module):
    def __init__(
        self, 
        grid_size: List[int],
        embed_dim: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool, 
        ckpt: bool,
        return_intermediate: bool = False) -> None:
        super().__init__()
        head_dim = embed_dim // head_num
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2
        self.grid_size = grid_size # H W
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        y, x = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing='ij')
        ids = torch.stack((x, y)).flatten(1).permute(1, 0).contiguous()
        emb = torch.cat([rope(ids[:, i], inv_freq) for i in range(ids.shape[-1])], dim=-3)
        self.register_buffer("pe", emb[None], persistent=False)

        self.blocks = nn.ModuleList([RoBlock(embed_dim, head_num, dropout, bias, flash_attn) for _ in range(block_num)])
        if not return_intermediate:
            self.norm = LayerNorm(embed_dim, bias) 
        else:
            self.norm = nn.ModuleList([LayerNorm(embed_dim, bias) for _ in range(block_num)])
        self.embed_dim = embed_dim
        self.ckpt = ckpt
        self.return_intermediate = return_intermediate

    def forward(self, x: Tensor) -> Tensor:
        if not self.return_intermediate:
            for block in self.blocks:
                if self.ckpt and self.training:
                    x = checkpoint.checkpoint(block, x, self.pe, use_reentrant=False)
                else:
                    x = block(x, self.pe)
            return self.norm(x)
        else:
            xs = []
            for i, block in enumerate(self.blocks):
                if self.ckpt and self.training:
                    x = checkpoint.checkpoint(block, x, self.pe, use_reentrant=False)
                else:
                    x = block(x, self.pe)
                xs.append(self.norm[i](x))
            return xs
        

class RoFormerPE3D(nn.Module):
    def __init__(
        self, 
        grid_size: List[int],
        embed_dim: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool, 
        ckpt: bool) -> None:
        super().__init__()
        head_dim = embed_dim // head_num
        assert head_dim % 6 == 0
        assert grid_size[0] == head_num
        self.head_dim = head_dim
        self.axis_dim = head_dim // 3
        self.grid_size = grid_size # H W
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        z, y, x = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), torch.arange(grid_size[2]), indexing='ij')
        ids = torch.stack((x, y, z)).flatten(1).permute(1, 0).contiguous()
        emb = torch.cat([rope(ids[:, i], inv_freq) for i in range(ids.shape[-1])], dim=-3)
        self.register_buffer("pe", emb[None].view(head_num, -1, head_dim // 2, 2, 2), persistent=False)

        self.blocks = nn.ModuleList([RoBlock(embed_dim, head_num, dropout, bias, flash_attn) for _ in range(block_num)])
        self.norm = LayerNorm(embed_dim, bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.ckpt and self.training:
                x = checkpoint.checkpoint(block, x, self.pe, use_reentrant=False)
            else:
                x = block(x, self.pe)
        return self.norm(x)
    
