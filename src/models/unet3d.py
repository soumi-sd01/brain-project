
import torch
import torch.nn as nn


# ==========================================
# 1. Double Convolution Block
# ==========================================

class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv3D, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(out_channels, affine=True),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(out_channels, affine=True),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


# ==========================================
# 2. Downsampling Block (Encoder)
# ==========================================

class Down3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Down3D, self).__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool3d(kernel_size=2),

            DoubleConv3D(in_channels, out_channels)

        )

    def forward(self, x):

        return self.maxpool_conv(x)


# ==========================================
# 3. Upsampling Block (Decoder)
# ==========================================

class Up3D(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):

        super(Up3D, self).__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)




class UNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=4):

        super(UNet3D, self).__init__()

        # Encoder
        self.inc = DoubleConv3D(in_channels, 32)

        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        self.down4 = Down3D(256, 512)

        # Decoder
        self.up1 = Up3D(512, 256, 256)
        self.up2 = Up3D(256, 128, 128)
        self.up3 = Up3D(128, 64, 64)
        self.up4 = Up3D(64, 32, 32)

        # Final segmentation layer
        self.outc = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Segmentation output
        return self.outc(x)
