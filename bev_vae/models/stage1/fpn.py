import torch.nn as nn


def norm(ch):
    return nn.GroupNorm(32, ch)


class IFPN(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        out_ch: int) -> None:
        super().__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2), 
            norm(in_ch // 2), nn.SiLU(),
            nn.ConvTranspose2d(in_ch // 2, in_ch // 4, 2, 2),
            norm(in_ch // 4), nn.SiLU(),
            nn.Conv2d(in_ch // 4, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1))
        
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2), 
            norm(in_ch // 2), nn.SiLU(),
            nn.Conv2d(in_ch // 2, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1))
        
        self.fpn3 = nn.Sequential(
            norm(in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1))
        
    def forward(self, x):
        fpns = []
        fpns.append(self.fpn1(x))
        fpns.append(self.fpn2(x))
        fpns.append(self.fpn3(x))
        return fpns
    

class SFPN(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        out_ch: int) -> None:
        super().__init__()
        self.fpn1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch * 2, 2, 2), 
            norm(in_ch * 2), nn.SiLU(),
            nn.Conv3d(in_ch * 2, in_ch * 4, 2, 2),
            norm(in_ch * 4), nn.SiLU(),
            nn.Conv3d(in_ch * 4, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1))
        
        self.fpn2 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch * 2, 2, 2), 
            norm(in_ch * 2), nn.SiLU(),
            nn.Conv3d(in_ch * 2, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1))
        
        self.fpn3 = nn.Sequential(
            norm(in_ch), nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, 1),
            norm(out_ch), nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1))
        
    def forward(self, x):
        fpns = []
        fpns.append(self.fpn1(x))
        fpns.append(self.fpn2(x))
        fpns.append(self.fpn3(x))
        return fpns