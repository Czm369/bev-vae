from pathlib import Path

from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

from bev_vae.data.transforms import denormalize


class ImageGeneration(Callback):
    def __init__(self, save_dir, row, interval, src_path=False):
        self.save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        self.row = row
        self.interval = interval
        self.src_path = src_path
        self.keys = [
            "inputs", "recons", "targets", "recons_i", 
            "recons_l", "recons_r", "recons_f", "recons_b",
            "gen", "ego_cond_gen", "low_res_recons", "super_res_gen"]
        for sclae in range(10):
            self.keys.append(f"cond_gen_{sclae}")
        for angle in range(0, 360, 30):
            self.keys.append(f"recons_rot_{angle}")
        for dx in [4, 2, 1, -1, -2, -4]:
            self.keys.append(f"recons_dx_{dx}")
        for dy in [4, 2, 1, -1, -2, -4]:
            self.keys.append(f"recons_dy_{dy}")
        self.conditional_keys = ["cond_da", "cond_cuboid", "cond_lane", "cond_pc"]
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.interval == 0:
            self.process_batch(trainer, pl_module, batch, "train", batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.process_batch(trainer, pl_module, batch, "val", batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.interval == 0:
            self.process_batch(trainer, pl_module, batch, "test", batch_idx)
    
    def process_batch(self, trainer, pl_module, batch, phase, batch_idx):
        logger = pl_module.logger.experiment
        if self.src_path:
            feats = pl_module.save_images(batch)
        else:
            feats = pl_module.log_images(batch)
        cam_num = len(batch['cam_names'])
        images = {k: denormalize(feats[k]) for k in self.keys if k in feats}
        global_step = trainer.global_step if phase != "test" else batch_idx
        if phase != "test" or batch_idx % self.interval == 0:
            for k in images:
                logger.add_image(f"{phase}/{k}", make_grid(images[k][:cam_num*self.row], nrow=cam_num), global_step=global_step)

            for k in ['img_feats_pre', 'img_feats_post']:
                if k in feats:
                    logger.add_image(f"{phase}/{k}", make_grid(feats[k][:cam_num*self.row], nrow=cam_num), global_step=global_step)

            for k in ['scn_feats_pre', 'scn_feats_post', 'state']:
                if k in feats:
                    logger.add_image(
                        f"{phase}/{k}", make_grid(feats[k][:self.row].permute(0, 1, 3, 2).flip(-1).flip(-2), nrow=self.row), global_step=global_step)

            if "condition" in feats:
                logger.add_image(
                    f"{phase}/condition", make_grid(feats["condition"][:self.row].permute(0, 1, 3, 2).flip(-1).flip(-2), nrow=self.row), global_step=global_step)
                for k in self.conditional_keys:
                    if k in feats:
                        logger.add_image(
                            f"{phase}/{k}", make_grid(feats[k][:self.row].permute(0, 1, 3, 2).flip(-1).flip(-2), nrow=self.row), global_step=global_step)
            
            for k in ["gt_map", "pd_map"]:
                if k in feats:
                    logger.add_image(
                        f"{phase}/{k}", make_grid(feats[k][:self.row].permute(0, 1, 3, 2).flip(-1).flip(-2), nrow=self.row), global_step=global_step)
                        
            for k in ['imgs_gt_bboxes', 'imgs_pd_bboxes']:
                if k in feats:
                    logger.add_image(f"{phase}/{k}", make_grid(feats[k][:cam_num*self.row], nrow=cam_num), global_step=global_step)

