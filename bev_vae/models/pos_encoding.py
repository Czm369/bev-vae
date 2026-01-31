import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PosEncoding1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.org_ch = ch
        self.ch = int(np.floor(ch / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.ch, 2).float() / self.ch))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("pe", None, persistent=False)

    def forward(self, x):
        if self.pe is not None and self.pe.shape == x.shape:
            return self.pe.to(x.dtype)
        B, T, E = x.shape
        pos_x = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((T, E), device=x.device, dtype=x.dtype)
        emb[:, :self.ch] = emb_x
        self.pe = emb[None].repeat(B, 1, 1)
        return self.pe.to(x.dtype)


class PosEncoding2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.org_ch = ch
        self.ch = int(np.floor(ch / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.ch, 2).float() / self.ch))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("pe", None, persistent=False)

    def forward(self, x):
        if self.pe is not None and self.pe.shape == x.shape:
            return self.pe.to(x.dtype)
        B, H, W, E = x.shape
        pos_x = torch.arange(W, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(H, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x)[None].repeat(H, 1, 1)
        emb_y = get_emb(sin_inp_y)[:, None].repeat(1, W, 1)
        emb = torch.zeros((H, W, E), device=x.device, dtype=x.dtype)
        emb[:, :, :self.ch] = emb_x
        emb[:, :, self.ch: self.ch * 2] = emb_y
        self.pe = emb[None].repeat(B, 1, 1, 1)
        return self.pe.to(x.dtype)


class PosEncoding3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.org_ch = ch
        self.ch = int(np.floor(ch / 6) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.ch, 2).float() / self.ch))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("pe", None, persistent=False)

    def forward(self, x):
        if self.pe is not None and self.pe.shape == x.shape:
            return self.pe.to(x.dtype)
        B, D, H, W, E = x.shape
        pos_x = torch.arange(W, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(H, device=x.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(D, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)[None, None].repeat(D, H, 1, 1)
        emb_y = get_emb(sin_inp_y)[None, :, None].repeat(D, 1, W, 1)
        emb_z = get_emb(sin_inp_z)[:, None, None].repeat(1, H, W, 1)
        emb = torch.zeros((D, H, W, E), device=x.device, dtype=x.dtype)
        emb[:, :, :, : self.ch] = emb_x
        emb[:, :, :, self.ch: self.ch * 2] = emb_y
        emb[:, :, :, self.ch * 2: self.ch * 3] = emb_z
        self.pe = emb[None].repeat(B, 1, 1, 1, 1)
        return self.pe.to(x.dtype)
