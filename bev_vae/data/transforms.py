from typing import Sequence

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (Compose, GaussianBlur, Normalize,
                                    RandomApply, RandomHorizontalFlip, Resize)

# Use timm's names
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)


def normalize(type: str = "IMAGENET") -> Normalize:
    if type == "IMAGENET":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif type == "SD":
        mean = std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("Only support IMAGENET and SD")
    return Normalize(mean=mean, std=std)

def denormalize(tensor: torch.Tensor, type: str = "IMAGENET") -> torch.Tensor:
    if type == "IMAGENET":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif type == "SD":
        mean = std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("Only support IMAGENET and SD")
    tensor = tensor.clone() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean).clamp_(0, 1)
    return tensor

def denormalize_z(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    tensor = tensor.clone() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


class RecordCompose(Compose):
    def __call__(self, img):
        matrix = torch.eye(3)
        for t in self.transforms:
            img = t(img)
            matrix = t.matrix @ matrix
        return img, matrix

class RResize(Resize):
    matrix = torch.eye(3)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        # self.size(h, w)
        w, h = img.size # PIL
        self.matrix = torch.diag(torch.tensor([self.size[1]/w, self.size[0]/h, 1.]))
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
    
class RRandomHorizontalFlip(RandomHorizontalFlip):
    matrix = torch.eye(3)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            self.matrix = torch.tensor([[-1., 0., img.size[0]], [0., 1., 0.], [0., 0., 1.]])
            return F.hflip(img)
        self.matrix = torch.eye(3)
        return img

class RandomGaussianBlur(RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)