import torch
import torch.nn as nn
from utils import center_crop_tensor

class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        # Encoder network
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.decoder1 = DecoderBlock(1024)
        self.decoder2 = DecoderBlock(512)
        self.decoder3 = DecoderBlock(256)
        self.decoder4 = DecoderBlock(128)
        # final segmentation map
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, s4)
        x = self.decoder2(x, s3)
        x = self.decoder3(x, s2)
        x = self.decoder4(x, s1)
        x = self.final_conv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    
    def forward(self, x):
        x = self.convolutions(x)
        skip = x
        x = self.downsample(x)
        return skip, x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = center_crop_tensor(x.shape, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.convolutions(x)
        return x

def unet():
    model = Unet()
    return model