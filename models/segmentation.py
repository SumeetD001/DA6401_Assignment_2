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

        _vgg = VGG11Encoder(num_classes=37)
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location="cpu")
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
        return self.final_conv(d1)
