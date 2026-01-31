import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, 
        image_encoder: nn.Module,
        scene_encoder: nn.Module, 
        state_encoder: nn.Module) -> None:
        super().__init__()
        self.image = image_encoder
        self.scene = scene_encoder
        self.state = state_encoder
        
    def forward(self, x, img_metas):
        return self.state(self.scene(self.image(x), img_metas))
    
    def get_img(self, x):
        return self.image(x)
    
    def get_scene(self, x, img_metas):
        return self.scene(self.image(x), img_metas).contiguous()

class Decoder(nn.Module):
    def __init__(
        self, 
        state_decoder: nn.Module,
        scene_decoder: nn.Module, 
        image_decoder: nn.Module) -> None:
        super().__init__()
        self.state = state_decoder
        self.scene = scene_decoder
        self.image = image_decoder
        
    def forward(self, x, img_metas):
        return self.image(self.scene(self.state(x), img_metas))
