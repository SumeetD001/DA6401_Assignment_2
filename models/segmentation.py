"""
U-Net Style Semantic Segmentation
==================================
Encoder : VGG11 convolutional backbone (skip connections retained)
Decoder : Symmetric expansive path with TransposedConv upsampling
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=3, backbone_weights=None, freeze_backbone=False):
        super().__init__()
        self.num_classes = num_classes

        # ---- Build encoder from VGG11 --------------------------------
        _vgg = VGG11Encoder(num_classes=37)
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location="cpu")
            # Strip ClassificationModel wrapper prefix if present
            state = {k.replace("model.", ""): v for k, v in state.items()}
            missing, unexpected = _vgg.load_state_dict(state, strict=False)
            feat_loaded = [k for k in state if k.startswith("features.") and k not in missing]
            print(f"[SegmentationModel] Loaded {len(feat_loaded)} backbone keys from {backbone_weights}")
            if len(feat_loaded) == 0:
                raise RuntimeError(
                    f"No backbone keys loaded from {backbone_weights}! "
                    "Check checkpoint was saved from ClassificationModel or VGG11."
                )

        feats = _vgg.features
        self.enc1  = feats[0]
        self.pool1 = feats[1]
        self.enc2  = feats[2]
        self.pool2 = feats[3]
        self.enc3  = nn.Sequential(feats[4], feats[5])
        self.pool3 = feats[6]
        self.enc4  = nn.Sequential(feats[7], feats[8])
        self.pool4 = feats[9]
        self.enc5  = nn.Sequential(feats[10], feats[11])
        self.pool5 = feats[12]

        if freeze_backbone:
            for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in m.parameters():
                    p.requires_grad = False

        self.bottleneck = _double_conv(512, 1024)
        self.up5 = _UpBlock(1024, 512, 512)
        self.up4 = _UpBlock(512,  512, 256)
        self.up3 = _UpBlock(256,  256, 128)
        self.up2 = _UpBlock(128,  128,  64)
        self.up1 = _UpBlock(64,    64,  64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        for m in [self.bottleneck, self.up5, self.up4,
                  self.up3, self.up2, self.up1, self.final_conv]:
            for layer in m.modules() if hasattr(m, "modules") else [m]:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @property
    def use_grad_checkpoint(self):
        return getattr(self, '_use_grad_checkpoint', False)

    @use_grad_checkpoint.setter
    def use_grad_checkpoint(self, value):
        self._use_grad_checkpoint = value

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        def _run(module, inp):
            if self.use_grad_checkpoint and inp.requires_grad:
                return grad_ckpt(module, inp, use_reentrant=False)
            return module(inp)

        s1 = _run(self.enc1, x)
        s2 = _run(self.enc2, self.pool1(s1))
        s3 = _run(self.enc3, self.pool2(s2))
        s4 = _run(self.enc4, self.pool3(s3))
        s5 = _run(self.enc5, self.pool4(s4))

        b  = self.bottleneck(self.pool5(s5))
        d5 = self.up5(b,  s5)
        d4 = self.up4(d5, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        return self.final_conv(d1)    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UpBlock(nn.Module):
    """Transposed-conv upsample → concat skip → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims in case of odd input sizes
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="nearest"
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegmentationModel(nn.Module):
    """
    VGG11-encoder U-Net for semantic segmentation.

    Parameters
    ----------
    num_classes       : int   Output channels (3 for Oxford-IIIT Pet trimap).
    backbone_weights  : str   Path to VGG11 state-dict (optional).
    freeze_backbone   : bool  Freeze encoder weights.
    """

    def __init__(
        self,
        num_classes: int = 3,
        backbone_weights: str = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ---- Build encoder from VGG11 --------------------------------
        _vgg = VGG11(num_classes=37)
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location="cpu")
            remapped = {}
            for k, v in state.items():
                new_key = k.replace("model.", "")
                remapped[new_key] = v
            _vgg.load_state_dict(remapped, strict=False)

        # Expose each block separately to capture skip-connection outputs
        feats = _vgg.features  # the big Sequential
        # VGG11 feature block indices (0-indexed items in feats):
        #   0: Conv-BN-ReLU(3→64)   1: MaxPool   → enc1 out before pool
        #   2: Conv-BN-ReLU(64→128) 3: MaxPool   → enc2
        #   4-5: 2× Conv(128→256)  6: MaxPool   → enc3
        #   7-8: 2× Conv(256→512)  9: MaxPool   → enc4
        #  10-11: 2× Conv(512→512) 12: MaxPool  → enc5

        # We wrap each "stage" (up to but NOT including MaxPool) separately.
        self.enc1 = feats[0]          # Conv 3→64
        self.pool1 = feats[1]
        self.enc2 = feats[2]          # Conv 64→128
        self.pool2 = feats[3]
        self.enc3 = nn.Sequential(feats[4], feats[5])   # Conv 128→256 ×2
        self.pool3 = feats[6]
        self.enc4 = nn.Sequential(feats[7], feats[8])   # Conv 256→512 ×2
        self.pool4 = feats[9]
        self.enc5 = nn.Sequential(feats[10], feats[11]) # Conv 512→512 ×2
        self.pool5 = feats[12]

        if freeze_backbone:
            for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in m.parameters():
                    p.requires_grad = False

        # ---- Bottleneck ---------------------------------------------
        self.bottleneck = _double_conv(512, 1024)

        # ---- Decoder ------------------------------------------------
        # Each _UpBlock: (in_ch from below, skip_ch from encoder, out_ch)
        self.up5 = _UpBlock(1024, 512, 512)
        self.up4 = _UpBlock(512,  512, 256)
        self.up3 = _UpBlock(256,  256, 128)
        self.up2 = _UpBlock(128,  128, 64)
        self.up1 = _UpBlock(64,    64, 64)

        # ---- Final 1×1 conv → class logits -------------------------
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_decoder()

    # ------------------------------------------------------------------
    def _init_decoder(self):
        for m in [self.bottleneck, self.up5, self.up4,
                  self.up3, self.up2, self.up1, self.final_conv]:
            for layer in (m.modules() if hasattr(m, "modules") else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, 224, 224)

        Returns
        -------
        (N, num_classes, 224, 224) logits (no softmax applied).
        """
        # Encoder
        s1 = self.enc1(x)          # (N, 64,  224, 224)
        s2 = self.enc2(self.pool1(s1))   # (N, 128, 112, 112)
        s3 = self.enc3(self.pool2(s2))   # (N, 256,  56,  56)
        s4 = self.enc4(self.pool3(s3))   # (N, 512,  28,  28)
        s5 = self.enc5(self.pool4(s4))   # (N, 512,  14,  14)

        # Bottleneck
        b = self.bottleneck(self.pool5(s5))  # (N, 1024, 7, 7)

        # Decoder (each stage doubles spatial resolution)
        d5 = self.up5(b,  s5)   # (N, 512,  14, 14)
        d4 = self.up4(d5, s4)   # (N, 256,  28, 28)
        d3 = self.up3(d4, s3)   # (N, 128,  56, 56)
        d2 = self.up2(d3, s2)   # (N, 64,  112, 112)
        d1 = self.up1(d2, s1)   # (N, 64,  224, 224)

        return self.final_conv(d1)  # (N, num_classes, 224, 224)
