import torch
import torch.nn as nn
from torchvision.models import resnet50

class Unet_Resnet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        resnet = resnet50(pretrained=True)
        # freeze pretrained encoder
        for param in resnet.parameters():
            param.requires_grad = False

        # encoder network (resnet50 backbone) (256x256)
        # 3 channels -> 64 channels (128x128)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])
        # 64 channels -> 256 channels (64x64)
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])
        # 256 channels -> 512 channels (32x32)
        self.encoder3 = nn.Sequential(*list(resnet.children())[5:6])
        # 512 channels -> 1024 channels (16x16)
        self.encoder4 = nn.Sequential(*list(resnet.children())[6:7])
        # 1024 channels -> 2048 channels(8x8)
        self.encoder5 = nn.Sequential(*list(resnet.children())[7:8])

        # bottleneck
        # 2048 channels -> 4096 channels (4x4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            ResidualBlock(4096)
        )

        # decoder network
        # 4096 channels -> 2048 channels (8x8)
        self.decoder1 = DecoderBlock(4096, 2048)
        # 2048 channels -> 1024 channels (16x16)
        self.decoder2 = DecoderBlock(2048, 1024)
        # 1024 channels -> 512 channels (32x32)
        self.decoder3 = DecoderBlock(1024, 512)
        # 512 channels -> 256 channels (64x64)
        self.decoder4 = DecoderBlock(512, 256)
        # 256 channels -> 64 channels (128x128)
        self.decoder5 = DecoderBlock(256, 64)
        # 64 channels -> 32 channels (256x256)
        self.decoder6 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32)
        )

        # final layer
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = s1 = self.encoder1(x)
        x = s2 = self.encoder2(x)
        x = s3 = self.encoder3(x)
        x = s4 = self.encoder4(x)
        x = s5 = self.encoder5(x)

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
        

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)
        x = self.residual_blocks(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connection = x
        residual = self.residual(x)
        out = residual + skip_connection
        return out
    
def unet_resnet50():
    model = Unet_Resnet50()
    return model