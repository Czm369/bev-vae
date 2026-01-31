import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule

from bev_vae.models.distributions import DiagonalGaussianDistribution


class BEVVAE(LightningModule):
    def __init__(
        self,
        image_key: str, 
        embed_dim: int,
        aug_extrinsic_prob: float,
        aug_size: List[float],
        drop_view_prob: float,
        load_ema: bool,
        monitor: str,
        encoder: nn.Module, 
        decoder: nn.Module, 
        loss: nn.Module, 
        evaluation: nn.Module,
        matcher: nn.Module,
        base_lr: float,
        scheduler = None,
        ckpt_path = None, 
        save_bev = False,
        bev_path = None,
        ignore_keys = []) -> None:
        super().__init__()
        
        self.save_bev = save_bev
        self.bev_path = Path(bev_path) if bev_path else bev_path
        self.image_key = image_key
        self.embed_dim = embed_dim
        self.aug_extrinsic_prob = aug_extrinsic_prob
        self.aug_size = aug_size
        self.drop_view_prob = drop_view_prob
        self.load_ema = load_ema
        self.monitor = monitor
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.evaluation = evaluation
        if self.evaluation is not None:
            self.evaluation.eval()
        self.matcher = matcher
        if self.matcher is not None:
            self.matcher.eval()
        self.base_lr = base_lr
        self.scheduler = scheduler
        self.pre_quant = nn.Linear(encoder.state.transformer.embed_dim, embed_dim*2)
        self.post_quant = nn.Linear(embed_dim, decoder.state.transformer.embed_dim)
        self.register_buffer(
            "img_color", 
            torch.randn(3, encoder.image.transformer.embed_dim, 1, 1))
        self.register_buffer(
            "scn_color", 
            torch.randn(3, encoder.scene.transformer.embed_dim * encoder.scene.scene_size[0], 1, 1))
        self.register_buffer(
            "stt_color", 
            torch.randn(3, self.embed_dim, 1, 1))
        self.register_buffer("cond_color", torch.randn(3, 4, 1, 1))  
        self.params = (
            list(self.encoder.parameters())+
            list(self.decoder.parameters())+
            list(self.pre_quant.parameters())+
            list(self.post_quant.parameters()))
        if getattr(self.loss, "discriminator", None):
            self.params += list(self.loss.discriminator.parameters())

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        if self.load_ema and 'optimizer_states' in ckpt:
            print("Load EMA params for BEV-VAE.")
            ema_params = ckpt['optimizer_states'][0]['ema']
            for i in range((len(self.params))):
                self.params[i].data.copy_(ema_params[i])
        print(f"Restored from {path}")

    def encode(self, x: torch.FloatTensor, img_metas: dict) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        state = self.encoder(x, img_metas)
        moments = self.pre_quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, x: torch.FloatTensor, img_metas: dict) -> torch.FloatTensor:
        state = self.post_quant(x)
        dec = self.decoder(state, img_metas)
        return dec
    
    def encode_code(self, x: torch.FloatTensor, img_metas: dict) -> torch.LongTensor:
        state = self.encoder(x, img_metas)
        moments = self.pre_quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        code = posterior.sample()
        return code

    def forward(self, image: torch.FloatTensor, img_metas: dict, sample_posterior: bool=True) -> torch.FloatTensor: 
        posterior = self.encode(image, img_metas)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z, img_metas)
        return dec, posterior

    def combine_all_views(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return rearrange(image, 'b v ... -> (b v) ...')
    
    def combine_all_series(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return rearrange(image, 'b s v ... -> (b s v) ...')

    def get_input(self, batch: Tuple[Any, Any], key: str) -> Any:
        x = batch[key] 
        if len(x.shape) == 3:
            return x.unsqueeze(1) # B1HW
        elif len(x.shape) == 5:
            return self.combine_all_views(x) # (BV)CHW
        elif len(x.shape) == 6:
            return self.combine_all_series(x) # (BSV)CHW
        else:
            return x # BCHW

    def test_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        return dict()

    def get_last_layer(self) -> nn.Parameter:
        return self.decoder.image.to_pixel.proj[-1].weight

    def cond2rgb(self, cond):
        dim = cond.shape[1]
        mask = 2 ** torch.arange(dim)
        bits = ((torch.arange(2 ** dim)[..., None].int() & mask) != 0).float()
        bits = bits * 2 - 1.
        bits = bits.T.reshape(-1, dim, dim)
        _, _, H, W = cond.shape
        cond[:, :, H//2 - 2 : H//2 + 2, W//2 - 2 : W//2 + 2] = bits
        x = F.conv2d(cond, weight=self.cond_color) 
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        # x[:, :, 62:66, 62:66] = -1
        return x

    def img2rgb(self, img_feats):
        x = F.conv2d(img_feats, weight=self.img_color)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def scn2rgb(self, scn_feats):
        x = rearrange(scn_feats, "b e d h w -> b (d e) h w")
        x = F.conv2d(x, weight=self.scn_color)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def stt2rgb(self, stt_feats):
        x = rearrange(stt_feats, "b (h w) e -> b e h w", h=self.decoder.state.grid_size[1])
        x = F.conv2d(x, weight=self.stt_color)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    
    def rotate_extrinsic(self, matrix, degree):
        theta = torch.tensor(degree / 180. * torch.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation = torch.tensor([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]], device=matrix.device)
        return rotation @ matrix
        
    def translate_extrinsic(self, matrix, dx, dy, dz):
        translation = torch.tensor([
            [1., 0, 0, dx],
            [0, 1., 0, dy],
            [0, 0, 1., dz],
            [0, 0, 0, 1.]], device=matrix.device)
        return translation @ matrix
    
    def forward_extrinsic(self, matrix, cam, d):
        b, c, _, _ = matrix.shape
        vector = matrix[..., :3, 2]
        translation = torch.eye(4, device=matrix.device)[None, None].repeat(b, c, 1, 1)
        translation[..., :3, 3] = vector[:, cam:cam+1] * d
        return translation @ matrix
    
    @torch.no_grad()
    def log_images(self, batch: Tuple[Any, Any]) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        img_metas = {k: batch[k] for k in ['geometric', 'intrinsic', 'extrinsic']}
        img_metas['cam_num'] = len(batch['cam_names'])
        img_metas['img_shape'] = x.shape[-2:]

        img_feats_pre = self.encoder.image(x[:, -3:])
        scn_feats_pre = self.encoder.scene(img_feats_pre, img_metas)
        stt_feats_pre = self.encoder.state(scn_feats_pre)
        moments = self.pre_quant(stt_feats_pre)
        posterior = DiagonalGaussianDistribution(moments)
        quant = posterior.sample()
        stt_feats_post = self.post_quant(quant)
        scn_feats_post = self.decoder.state(stt_feats_post)
        img_feats_post = self.decoder.scene(scn_feats_post, img_metas)
        dec = self.decoder.image(img_feats_post)

        log["inputs"] = x[:, -3:].clone()
        log["targets"] = x[:, :3].clone()
        log["recons"] = dec

        log["img_feats_pre"] = self.img2rgb(img_feats_pre)
        log["img_feats_post"] = self.img2rgb(img_feats_post)
        log["scn_feats_pre"] = self.scn2rgb(scn_feats_pre)
        log["scn_feats_post"] = self.scn2rgb(scn_feats_post)
        log["state"] = self.stt2rgb(quant)

        return log
