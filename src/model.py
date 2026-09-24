"""Модели сегментации дефектов с выходом в виде логитов одного канала."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """Дважды применяет свёртку, нормализацию и ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Объединяет признаки декодера и пропущенного уровня энкодера."""

    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Явный размер убирает ошибки округления для сторон, не кратных 32.
        decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, decoder), dim=1))


class UNet(nn.Module):
    """Компактная U-Net, обучаемая с нуля."""

    def __init__(self) -> None:
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images shaped [N, 3, H, W].")

        x1 = self.enc1(images)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x = self.bottleneck(self.pool(x4))

        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        return self.head(self.up1(x, x1))


class ResNet18UNet(nn.Module):
    """U-Net с энкодером ResNet18 и собственным декодером."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("Install torchvision to use 'resnet18_unet'.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = resnet18(weights=weights)

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images shaped [N, 3, H, W].")

        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(images)))
        level1 = self.encoder.layer1(self.encoder.maxpool(stem))
        level2 = self.encoder.layer2(level1)
        level3 = self.encoder.layer3(level2)
        level4 = self.encoder.layer4(level3)

        x = self.up4(level4, level3)
        x = self.up3(x, level2)
        x = self.up2(x, level1)
        x = self.up1(x, stem)
        x = F.interpolate(x, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)


def build_model(name: str, pretrained: bool = False) -> nn.Module:
    """Создаёт модель по имени; pretrained относится только к ResNet18."""

    key = name.strip().lower().replace("-", "_")
    if key == "unet":
        if pretrained:
            raise ValueError("'unet' has no pretrained encoder; use 'resnet18_unet'.")
        return UNet()
    if key in {"resnet18_unet", "resnet18"}:
        return ResNet18UNet(pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}; choose 'unet' or 'resnet18_unet'.")
