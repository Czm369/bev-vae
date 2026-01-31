import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from bev_vae.data.datasets.argoverse_sensor import SensorDataset
from bev_vae.data.transforms import (RecordCompose, RRandomHorizontalFlip,
                                     RResize, normalize)


@dataclass
class Argoverse(Dataset):
    data_dir: str
    split: str
    cam_names: List[str]
    other_sensors: List[str]
    class_num: int 
    scene_size: List[int] 
    pc: List[float] 
    with_condition: bool = False
    return_bev: bool = False
    cam_res: Tuple[int, int] = (256, 256)
    augment: bool = False
    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.dataset = SensorDataset(
            data_dir=self.data_dir / 'sensor', 
            split=self.split, 
            cam_names=self.cam_names, 
            other_sensors=self.other_sensors, 
            class_num=self.class_num, 
            scene_size=self.scene_size, 
            pc=self.pc, 
            with_condition=self.with_condition,
            return_bev=self.return_bev)
        self.preprocess = T.ToPILImage() 
        self.resize = RecordCompose([RResize(self.cam_res)])
        self.geometric_augmentation = RecordCompose([RRandomHorizontalFlip(p=0.5)])
        self.color_jittering = T.Compose([T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8), 
            T.RandomGrayscale(p=0.2)])
        self.normalize = T.Compose([
            T.ToTensor(), normalize()])
        print(f'Argoverse {self.split} has {len(self)} samples.')

    def process_img(self, img):
        img, geometric = self.resize(self.preprocess(img)) 
        if self.augment:
            img_aug, geometric_aug = self.geometric_augmentation(img.copy())
            geometric = geometric_aug @ geometric
            img_aug = self.color_jittering(img_aug)
            return torch.cat([self.normalize(img), self.normalize(img_aug)]), geometric
        return self.normalize(img), geometric
    
    def get_geometric(self, w, h):
        return torch.diag(torch.tensor([self.cam_res[1]/w, self.cam_res[0]/h, 1.]))

    def __getitem__(self, idx: int):
        while True:
            try:
                datum = self.dataset[idx]
                if self.return_bev:
                    img_sizes = [(2048, 1550)] * len(self.cam_names) # av2
                    img_sizes[3] = (1550, 2048)
                    geometric = [self.get_geometric(w, h) for w, h in img_sizes]
                    intrinsic = [datum["latent"][cam]["intrinsic"] for cam in self.cam_names]
                    extrinsic = [datum["latent"][cam]["extrinsic"] for cam in self.cam_names]
                    datum["path"] =  datum["latent"]["path"]
                    datum["bev"] =  datum["latent"]["bev"]
                    del datum["latent"]
                else:
                    imgs, geometric = zip(*[self.process_img(datum["synchronized_imagery"][cam]["img"]) for cam in self.cam_names])
                    intrinsic = [datum["synchronized_imagery"][cam]["intrinsic"] for cam in self.cam_names]
                    extrinsic = [datum["synchronized_imagery"][cam]["extrinsic"] for cam in self.cam_names]
                    datum["paths"] =  [datum["synchronized_imagery"][cam]["path"] for cam in self.cam_names]
                    datum["imgs"] = torch.stack(imgs) 
                    del datum["synchronized_imagery"]

                datum["geometric"] = torch.stack(geometric)
                datum["intrinsic"] = np.stack(intrinsic).astype(np.float32) # float64 -> float32
                datum["extrinsic"] = np.stack(extrinsic).astype(np.float32) # float64 -> float32
                datum["cam_names"] = self.cam_names
                datum["sample_token"] = f'{datum["log_id"]}_{datum["timestamp_ns"]}'
                return datum
            except Exception as e:
                print(f"Error accessing index {idx}: {e}. Picking a new index.")
                idx = random.randint(0, len(self.dataset) - 1)

    def __len__(self):
        return len(self.dataset)
