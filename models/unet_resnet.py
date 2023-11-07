import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils import center_crop_tensor

class Unet_Resnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        resnet = resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
            
        # resnet backbone encoder
        self.encoder1 = nn.Sequential(
            *list(resnet.children())[:3]
        )
        self.encoder2 = nn.Sequential(
            *list(resnet.children())[3:5]
        )
        self.encoder3 = nn.Sequential(
            *list(resnet.children())[5:6]
        )
        self.encoder4 = nn.Sequential(
            *list(resnet.children())[6:7]
        )
        self.encoder5 = nn.Sequential(
            *list(resnet.children())[7:8]
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # decoder network
        self.decoder1 = DecoderBlock(1024)
        self.decoder2 = DecoderBlock(512)
        self.decoder3 = DecoderBlock(256)
        self.decoder4 = DecoderBlock(128)
        self.decoder5 = DecoderBlock(64, halve=False)
        self.decoder6 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # final layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        x = self.encoder1(x)
        s1 = x
        x = self.encoder2(x)
        s2 = x
        x = self.encoder3(x)
        s3 = x
        x = self.encoder4(x)
        s4 = x
        x = self.encoder5(x)
        s5 = x
        x = self.bottleneck(x)
        x = self.decoder1(x, s5)
        x = self.decoder2(x, s4)
        x = self.decoder3(x, s3)
        x = self.decoder4(x, s2)
        x = self.decoder5(x, s1)
        x = self.decoder6(x)
        x = self.final_conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, halve=True):
        super().__init__()
        if halve:
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
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU()
            )
        else:
            self.upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
            self.convolutions = nn.Sequential(
                nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = center_crop_tensor(x.shape, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.convolutions(x)
        return x
    
def unet_resnet():
    model = Unet_Resnet()
    return model