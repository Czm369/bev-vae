from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from bev_vae.models.layers import MSDATransformer2D, MSDATransformer3D
from bev_vae.models.pos_encoding import PosEncoding3D
from bev_vae.models.stage1.fpn import IFPN, SFPN


class SceneEncoder(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: int,
        bias: bool,
        ckpt: bool,
        image_size: List[int],
        scene_size: List[int],
        pc: List[int],
        ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_size = image_size
        self.scene_size = scene_size
        self.pc = pc
        self.scene_embedding = nn.Embedding(scene_size[1] * scene_size[2], out_ch)
        self.pos_embedding = PosEncoding3D(out_ch)
        self.register_buffer("scene_3d", self.get_scene_point(*scene_size))
        self.fpn = IFPN(in_ch, out_ch)
        self.transformer = MSDATransformer2D(
            out_ch, block_num, head_num, level_num, point_num, dropout, bias, ckpt)
        
    @staticmethod
    def get_scene_point(D, H, W):
        z = torch.linspace(0.5, D - 0.5, D).view(D, 1, 1).repeat(1, H, W) / D
        y = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).repeat(D, 1, W) / H
        x = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).repeat(D, H, 1) / W
        point = torch.stack([x, y, z]).flatten(1).permute(1, 0).contiguous() # (DHW) xyz
        return point 

    def forward(self, x, img_metas):
        img_feats = self.fpn(x)
        img_feats =[rearrange(x, "(b v) ... -> b v ...", v = img_metas["cam_num"]) for x in img_feats]
        b = img_feats[0].shape[0]
        d, h, w = self.scene_size
        query = self.scene_embedding.weight.reshape(h, w, -1)[None, None].repeat(b, d, 1, 1, 1)
        query += self.pos_embedding(query) # B D H W E
        query = query.flatten(1, 3) # B (DHW) E

        shape, value = [], []
        for lvl, feat in enumerate(img_feats):
            shape.append(feat.shape[-2:])
            feat = feat.flatten(3).permute(1, 0, 3, 2) # B V E H W -> B V E (HW) -> V B (HW) E
            value.append(feat)
        shape = torch.tensor(shape, device=query.device)
        value = torch.cat(value, 2).permute(0, 2, 1, 3) # V B sum(HW) E -> V sum(HW) B E 

        point = self.scene_3d[None].repeat(b, 1, 1) # B (DHW) xyz
        point, valid = self.point_sampling(point, img_metas) # point: V B (DHW) xy , valid: V B (DHW)
        scene = self.transformer(query, value, point, valid, shape)

        return scene.permute(0, 2, 1).reshape(b, self.out_ch, *self.scene_size) # B E D H W

    def point_sampling(self, point, img_metas, eps=1e-6):
        _, dhw, _ = point.shape # B (DHW) 3
        point[..., 0:1] = point[..., 0:1] * (self.pc[3] - self.pc[0]) + self.pc[0]
        point[..., 1:2] = point[..., 1:2] * (self.pc[4] - self.pc[1]) + self.pc[1]
        point[..., 2:3] = point[..., 2:3] * (self.pc[5] - self.pc[2]) + self.pc[2]
        point = torch.cat([point, torch.ones_like(point[..., :1])], -1) # B (DHW) 4
        point = point[:, None].repeat(1, img_metas["cam_num"], 1, 1)[..., None] # B V (DHW) 4 1
        point = torch.inverse(img_metas["extrinsic"].float())[:, :, None].repeat(1, 1, dhw, 1, 1).to(point.dtype) @ point
        point = img_metas["intrinsic"][:, :, None].repeat(1, 1, dhw, 1, 1) @ point[..., :3, :]
        valid_z = point.squeeze(-1)[..., -1] > eps
        # Same geometrics in a batch is necessary, for valid points is same between different samples.
        point = img_metas["geometric"][:, :, None].repeat(1, 1, dhw, 1, 1) @ point
        point = point.squeeze(-1) # B V (DHW) 3
        # B V (DHW) 2 by x/z and y/z
        point = point[..., :2] / torch.maximum(point[..., -1:], torch.ones_like(point[..., -1:]) * eps)
        point[..., 0] /=  self.image_size[1] # x / W
        point[..., 1] /=  self.image_size[0] # y / H
        valid_x = (0. < point[..., 0]) & (point[..., 0] < 1.)
        valid_y = (0. < point[..., 1]) & (point[..., 1] < 1.)
        valid = valid_x & valid_y & valid_z
        point = point.permute(1, 0, 2, 3) # V B (DHW) 2
        valid = valid.permute(1, 0, 2) # V B (DHW)
        return point, valid # xy    


class SceneDecoder(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: int,
        bias: bool,
        ckpt: bool,
        image_size: List[int],
        frustum_size: List[int],
        pc: List[int],
        ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_size = image_size
        self.frustum_size = frustum_size
        self.pc = pc
        self.frustum_embedding = nn.Embedding(frustum_size[1] * frustum_size[2], in_ch)
        self.pos_embedding = PosEncoding3D(in_ch)
        self.register_buffer("frustum_3d", self.get_frustum_point(*frustum_size, image_size))
        self.fpn = SFPN(in_ch, in_ch)
        self.transformer = MSDATransformer3D(
            in_ch, block_num, head_num, level_num, point_num, frustum_size[0], dropout, bias, ckpt)
        self.proj = nn.Linear(in_ch, out_ch)
        
    @staticmethod
    def get_frustum_point(D, H, W, image_size):
        z = torch.linspace(0.5, D - 0.5, D).view(D, 1, 1).repeat(1, H, W)
        y = torch.linspace(0.5, image_size[0] - 0.5, H).view(1, H, 1).repeat(D, 1, W)
        x = torch.linspace(0.5, image_size[1] - 0.5, W).view(1, 1, W).repeat(D, H, 1)
        point = torch.stack([x * z, y * z, z]).flatten(1).permute(1, 0).contiguous() # (DHW) xyz
        return point
    
    def forward(self, x, img_metas):
        scn_feats = self.fpn(x)
        b = scn_feats[0].shape[0]
        d, h, w = self.frustum_size
        query = self.frustum_embedding.weight.reshape(h, w, -1)[None, None].repeat(b, d, 1, 1, 1)
        query += self.pos_embedding(query) # B D H W E
        query = query.flatten(1, 3)[:, None].repeat(1, img_metas["cam_num"], 1, 1) # B V (DHW) E
        # query += self.cam_embedding.weight[None, :, None, :]

        shape, value = [], []
        for lvl, feat in enumerate(scn_feats):
            shape.append(feat.shape[-3:])
            feat = feat.flatten(2).permute(0, 2, 1) # B E D H W -> B (DHW) E
            value.append(feat)
        shape = torch.tensor(shape, device=query.device)
        value = torch.cat(value, 1).permute(1, 0, 2) # B sum(DHW) E -> sum(DHW) B E

        point = self.frustum_3d[None].repeat(b, 1, 1) # B (DHW) xyz
        point, valid = self.point_sampling(point, img_metas) # point: V B (DHW) xyz , valid: V B (DHW)
        frustum = self.transformer(query, value, point, valid, shape).flatten(0, 1) # (BV) (HW) E
        frustum = rearrange(self.proj(frustum), "bv (h w) e -> bv e h w", h=h)
        return frustum

    def point_sampling(self, point, img_metas):
        _, dhw, _ = point.shape # B (DHW) 3
        point = point[:, None].repeat(1, img_metas["cam_num"], 1, 1)[..., None] # B V (DHW) 3 1
        point = torch.inverse(img_metas["geometric"].float())[:, :, None].repeat(1, 1, dhw, 1, 1).to(point.dtype) @ point
        point = torch.inverse(img_metas["intrinsic"].float())[:, :, None].repeat(1, 1, dhw, 1, 1).to(point.dtype) @ point
        point = point.squeeze(-1) # B V (DHW) 3 
        point = torch.cat([point, torch.ones_like(point[..., :1])], -1)[..., None] # B V (DHW) 4 1
        point = img_metas["extrinsic"][:, :, None].repeat(1, 1, dhw, 1, 1) @ point
        point = point.squeeze(-1)[..., :3] # B V (DHW) 3
        point[..., 0] = (point[..., 0] - self.pc[0]) / (self.pc[3] - self.pc[0])
        point[..., 1] = (point[..., 1] - self.pc[1]) / (self.pc[4] - self.pc[1])
        point[..., 2] = (point[..., 2] - self.pc[2]) / (self.pc[5] - self.pc[2])
        valid_x = (0. < point[..., 0]) & (point[..., 0] < 1.)
        valid_y = (0. < point[..., 1]) & (point[..., 1] < 1.)
        valid_z = (0. < point[..., 2]) & (point[..., 2] < 1.)
        valid = valid_x & valid_y & valid_z
        point = point.permute(1, 0, 2, 3) # V B (DHW) 3
        valid = valid.permute(1, 0, 2) # V B (DHW)
        return point, valid
