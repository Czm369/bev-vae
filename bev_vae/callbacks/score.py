from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from bev_vae.data.transforms import denormalize


class Score(Callback):
    def __init__(self, cache_size, epoch_interval=1, save_dir=None):
        self.save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        self.cache_size = cache_size
        self.epoch_interval = epoch_interval
        self.act1 = []
        self.act2 = dict()
        self.psnr = dict()
        self.ssim = dict()
        self.mvsc = dict()

    @torch.no_grad()
    def calculate_activation_statistics(self, model, images):
        model.eval()
        act = model(images)[0].squeeze(3).squeeze(2).cpu().numpy()
        return act

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean.real)
            else:
                fid = np.float64(-1)
        else:
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid

    def calculate_psnr(self, targets, images):
        return np.mean([psnr(targets[i], images[i], data_range=1.0) for i in range(len(targets))])

    def calculate_ssim(self, targets, images):
        return np.mean([ssim(targets[i], images[i], data_range=1.0, channel_axis=0) for i in range(len(targets))])
    
    @torch.no_grad()
    def calculate_mvsc(self, model, images):
        model.eval()
        correspondences = model(images)
        return correspondences['confidence'].mean().cpu().numpy()

    def recon_metrics(self, images, key, trainer, pl_module):
        targets = denormalize(images["targets"]).cpu().numpy()
        images = denormalize(images[key.split("/")[-1]])
        self.psnr[key].append(self.calculate_psnr(targets, images.cpu().numpy()))
        self.ssim[key].append(self.calculate_ssim(targets, images.cpu().numpy()))
        self.mvsc[key].append(self.calculate_mvsc(pl_module.matcher, images))
        if len(self.mvsc[key]) >= self.cache_size:
            pl_module.logger.experiment.add_scalar(key + "_psnr", np.mean(self.psnr[key]), trainer.global_step)
            pl_module.logger.experiment.add_scalar(key + "_ssim", np.mean(self.ssim[key]), trainer.global_step)
            pl_module.logger.experiment.add_scalar(key + "_mvsc", np.mean(self.mvsc[key]), trainer.global_step)

    def recon_fid(self, images, key, trainer, pl_module):
        images = denormalize(images[key.split("/")[-1]])
        self.act2[key].append(self.calculate_activation_statistics(pl_module.evaluation, images))
        if len(self.act2[key]) >= self.cache_size:
            act1, act2 = np.concatenate(self.act1), np.concatenate(self.act2[key])
            mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
            mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
            pl_module.logger.experiment.add_scalar(
                key + "_fid", 
                self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2), trainer.global_step)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.act1 = []
        self.act2 = dict()
        self.psnr = dict()
        self.ssim = dict()
        self.mvsc = dict()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.epoch_interval == 0:
            images = pl_module.log_images(batch)
            targets = denormalize(images["targets"])
            self.act1.append(self.calculate_activation_statistics(pl_module.evaluation, targets))
            if "val/targets" not in self.mvsc:
                self.mvsc["val/targets"] = []
            self.mvsc["val/targets"].append(self.calculate_mvsc(pl_module.matcher, targets))
            if len(self.mvsc["val/targets"]) >= self.cache_size:
                pl_module.logger.experiment.add_scalar("val/targets_mvsc", np.mean(self.mvsc["val/targets"]), trainer.global_step)
            keys = [
                "val/recons", "val/gen", "val/cond_gen", 
                "val/ego_cond_gen", "val/super_res_gen"
            ]
            for scale in range(10):
                keys.append(f"val/cond_gen_{scale}")
            for angle in range(0, 360, 30):
                keys.append(f"val/recons_rot_{angle}")
            for dx in [4, 2, 1, -1, -2, -4]:
                keys.append(f"val/recons_dx_{dx}")
            for dy in [4, 2, 1, -1, -2, -4]:
                keys.append(f"val/recons_dy_{dy}")

            for key in keys:
                if key.split("/")[-1] in images:
                    if key not in self.act2:
                        self.act2[key] = []
                        self.psnr[key] = []
                        self.ssim[key] = []
                        self.mvsc[key] = []
                    self.recon_fid(images, key, trainer, pl_module)
                    self.recon_metrics(images, key, trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self.act1 = []
        self.act2 = dict()
        self.psnr = dict()
        self.ssim = dict()
        self.mvsc = dict()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        images = pl_module.log_images(batch)
        targets = denormalize(images["targets"])
        self.act1.append(self.calculate_activation_statistics(pl_module.evaluation, targets))
        if "test/targets" not in self.mvsc:
            self.mvsc["test/targets"] = []
        self.mvsc["test/targets"].append(self.calculate_mvsc(pl_module.matcher, targets))
        if len(self.mvsc["test/targets"]) >= self.cache_size:
            pl_module.logger.experiment.add_scalar("test/targets_mvsc", np.mean(self.mvsc["test/targets"]), trainer.global_step)

        keys = [
            "test/recons", "test/gen", "test/cond_gen", 
            "test/ego_cond_gen", "test/super_res_gen"
        ]

        for scale in range(10):
            keys.append(f"test/cond_gen_{scale}")
        for angle in range(0, 360, 30):
            keys.append(f"test/recons_rot_{angle}")
        for dx in [4, 2, 1, -1, -2, -4]:
            keys.append(f"test/recons_dx_{dx}")
        for dy in [4, 2, 1, -1, -2, -4]:
            keys.append(f"test/recons_dy_{dy}")

        for key in keys:
            if key.split("/")[-1] in images:
                if key not in self.act2:
                    self.act2[key] = []
                    self.psnr[key] = []
                    self.ssim[key] = []
                    self.mvsc[key] = []
                self.recon_fid(images, key, trainer, pl_module)
                self.recon_metrics(images, key, trainer, pl_module)

 