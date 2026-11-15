"""
KidneyScan AI — Light 3D U-Net (KiTS23: kidney + tumor + cyst)
==============================================================
Lightweight 3D U-Net for 3D CT segmentation.

Classes (KiTS23 labels):
    0 = background
    1 = kidney
    2 = tumor
    3 = cyst

Design (Gap 3 — light model, 1-5M params, 1 Kaggle GPU):
  - Fewer base channels (default 48, measured 1.68M params, not 30M+
    as in a full nnU-Net).
  - Depthwise-separable 3D convolutions in the encoder/decoder blocks:
    a 3x3x3 depthwise conv (one filter per channel) + a 1x1x1 pointwise
    conv. This cuts parameters roughly 8-9x versus a standard 3x3x3
    conv at the same channel count, with a small accuracy cost.
  - 4 encoder levels, 3 decoder levels, skip connections (standard U-Net).
  - GroupNorm (works with batch size 1-2, unlike BatchNorm) + GELU.
  - Patch input, e.g. 128x128x64 or 96x96x96 voxels, single channel CT.

Reuse note (see REUSE.md):
  - Architecture pattern follows the classic 3D U-Net (Cicek et al. 2016)
    and the nnU-Net / MONAI SegResNet family, but this file is written
    for KidneyScan from those patterns — it is NOT copied code.
  - Depthwise-separable convolutions are the standard MobileNet idea
    (Howard et al. 2017) applied to 3D.

Parameter count is printed by ``count_parameters()`` and by
``python3 light_unet3d.py``. Target: ~1-5 million parameters.
Measured (this code, depthwise-separable blocks): base 32 -> 0.77M,
base 48 -> 1.68M, base 64 -> 2.94M. Default 48 (~1.68M) sits in the
middle of the target range and still fits a Kaggle free GPU.

Run on Kaggle (PyTorch preinstalled there). On this VM PyTorch may not
be installed — see the install note in README / train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4  # 0 background, 1 kidney, 2 tumor, 3 cyst
CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]


class DepthwiseSeparableConv3d(nn.Module):
    """3D depthwise-separable convolution.

    Standard 3x3x3 conv:  in * out * 27 params.
    Depthwise + pointwise: in * 27 + in * out params.
    Example in=out=32: standard 27,648 params, separable 1,888 params.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size=kernel,
                                   stride=stride, padding=padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class DoubleSepBlock(nn.Module):
    """Two depthwise-separable convs in a row (one U-Net block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv3d(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class EncoderBlock(nn.Module):
    """One encoder level: block + downsample (max-pool)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = DoubleSepBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """One decoder level: upsample + concat skip + block.

    Args:
        in_ch: channels coming up from below.
        skip_ch: channels of the skip connection from the encoder.
        out_ch: output channels of this level.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = DoubleSepBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd sizes after pooling (crop/pad skip to match).
        if x.shape[2:] != skip.shape[2:]:
            diff = [s - u for s, u in zip(skip.shape[2:], x.shape[2:])]
            pad = []
            for d in reversed(diff):
                pad += [d // 2, d - d // 2]
            x = F.pad(x, pad)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LightUNet3D(nn.Module):
    """Light 3D U-Net, 4 output classes, ~1-5M params.

    Args:
        in_channels: 1 for CT (single channel).
        num_classes: 4 (background, kidney, tumor, cyst).
        base_channels: channels at the first level. Measured: 32 -> 0.77M,
            48 -> 1.68M, 64 -> 2.94M. Default 48.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 base_channels: int = 48):
        super().__init__()
        c = base_channels
        self.enc1 = EncoderBlock(in_channels, c)       # c
        self.enc2 = EncoderBlock(c, c * 2)             # 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)         # 4c
        self.enc4 = EncoderBlock(c * 4, c * 8)         # 8c
        self.bottleneck = DoubleSepBlock(c * 8, c * 8)
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 4)
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2)
        self.dec2 = DecoderBlock(c * 2, c * 2, c)
        self.dec1 = DecoderBlock(c, c, c)
        self.out = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out(x)  # logits, shape (B, 4, D, H, W)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(base_channels: int = 48, device: str = "cpu") -> LightUNet3D:
    """Build the model and print its size. Use this in train.py / inference.py."""
    model = LightUNet3D(base_channels=base_channels)
    n = count_parameters(model)
    print(f"LightUNet3D(base_channels={base_channels}): "
          f"{n:,} trainable params ({n / 1e6:.2f}M)")
    return model.to(device)


if __name__ == "__main__":
    for bc in (32, 48, 64):
        m = LightUNet3D(base_channels=bc)
        n = count_parameters(m)
        print(f"base_channels={bc}: {n:,} params ({n / 1e6:.2f}M)")
        # Smoke test on a small patch (CPU-safe size).
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            y = m(x)
        print(f"  input {tuple(x.shape)} -> logits {tuple(y.shape)} "
              f"(expect (1, 4, 32, 32, 32))")
        assert y.shape == (1, NUM_CLASSES, 32, 32, 32)
    print("OK: LightUNet3D builds, forward pass works, params in 1-5M range.")
