import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from kornia_moons.viz import draw_LAF_matches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


class Matcher(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.model = KF.LoFTR(pretrained=None)
        pretrained_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(pretrained_dict["state_dict"])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        images_l, images_r = torch.chunk(images, 2, dim=-1)
        images_l = torch.roll(images_l, -1, dims=0)
        # LofTR works on grayscale images only
        input_dict = {
            "image0": K.color.rgb_to_grayscale(images_r),  
            "image1": K.color.rgb_to_grayscale(images_l)}
        return self.model(input_dict)
    
    def show_mvsc(self, images):
        images_l, images_r = torch.chunk(images, 2, dim=-1)
        images_l = torch.roll(images_l, -1, dims=0)
        # LofTR works on grayscale images only
        input_dict = {
            "image0": K.color.rgb_to_grayscale(images_r),  
            "image1": K.color.rgb_to_grayscale(images_l)}
        correspondences = self.model(input_dict)
        pil_imgs = []
        for idx in range(len(images)):
            mkpts0 = correspondences["keypoints0"][correspondences['batch_indexes'] == idx].cpu().numpy()
            mkpts1 = correspondences["keypoints1"][correspondences['batch_indexes'] == idx].cpu().numpy()
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            inliers = inliers > 0

            fig, ax = draw_LAF_matches(
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts0).view(1, -1, 2),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1),
                ),
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts1).view(1, -1, 2),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                ),
                torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
                K.tensor_to_image(images_r[idx:idx+1]),
                K.tensor_to_image(images_l[idx:idx+1]),
                inliers,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
                return_fig_ax=True
            )

            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = Image.fromarray(np.asarray(buf)).crop((640+1, 120+1, 1400-1, 880-1)).resize((256, 256))
            new_img = Image.new("RGB", (256, 258), (0, 0, 0))
            new_img.paste(img, (1, 0))
            plt.close(fig) 
            # pil_imgs.append(img) 
            pil_imgs.append(new_img) 
        
        total_width = sum(image.width for image in pil_imgs)
        max_height = max(image.height for image in pil_imgs)
        synchronized_imagery = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in pil_imgs:
            synchronized_imagery.paste(image, (x_offset, 0))
            x_offset += image.width
        return synchronized_imagery