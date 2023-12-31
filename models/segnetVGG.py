import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class SegnetVGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        # Encoder will use a pretrained VGG16 with bn to speed up convergence and training
        self.encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.decoder = nn.Sequential(
            UpsampleBlock(512, halve_channels=False),
            ConvBlock3(512),
            UpsampleBlock(512, halve_channels=False),
            ConvBlock3(512),
            UpsampleBlock(512),
            ConvBlock3(256),
            UpsampleBlock(256),
            ConvBlock2(128),
            UpsampleBlock(128),
            ConvBlock2(64)
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return(x)

class UpsampleBlock(nn.Module):
    # halves the number of channels to match vgg
    def __init__(self, in_channels, halve_channels=True):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        if halve_channels:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2), kernel_size=1)
        else:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return self.relu(x)


class ConvBlock3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.convolutions(x)
    
class ConvBlock2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.convolutions(x)
    
def Segnet_VGG():
    model = SegnetVGG()
    return model
