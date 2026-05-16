import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Padding keeps concatenation stable for odd input sizes.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net for binary green-area segmentation.

    The model returns raw logits with shape [B, 1, H, W].
    Use BCEWithLogitsLoss during training and sigmoid during inference.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: int = 64):
        super().__init__()
        self.input_block = DoubleConv(in_channels, features)
        self.down1 = DownBlock(features, features * 2)
        self.down2 = DownBlock(features * 2, features * 4)
        self.down3 = DownBlock(features * 4, features * 8)
        self.down4 = DownBlock(features * 8, features * 16)

        self.up1 = UpBlock(features * 16, features * 8, features * 8)
        self.up2 = UpBlock(features * 8, features * 4, features * 4)
        self.up3 = UpBlock(features * 4, features * 2, features * 2)
        self.up4 = UpBlock(features * 2, features, features)
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.input_block(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.output_conv(x)
