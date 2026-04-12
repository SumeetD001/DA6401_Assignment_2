"""
Multi-Task Perception Model
============================
Loads pre-trained weights from classifier.pth, localizer.pth, unet.pth
and exposes a single forward pass returning all three task outputs.
"""

import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import _double_conv, _UpBlock


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        classifier_path="classifier.pth",
        localizer_path="localizer.pth",
        unet_path="unet.pth",
        num_classes=37,
        seg_classes=3,
        device="cpu",
    ):
        super().__init__()

        import gdown
        gdown.download(id="1E_7yKvLmdlHbxzDnEHhkIY9GU8Ioe4O3", output=classifier_path, quiet=False)
        gdown.download(id="1FyyQPd19RUCg6jqCNPVTaeQEvJgJhDK6",  output=localizer_path,  quiet=False)
        gdown.download(id="1FyyQPd19RUCg6jqCNPVTaeQEvJgJhDK6",       output=unet_path,       quiet=False)

        self.device = torch.device(device)

        # ── Shared backbone: individual stages for skip connections ─────
        _base = VGG11Encoder(num_classes=num_classes)
        feats = _base.features

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
        self.avgpool = _base.avgpool

        # ── Classification head ─────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # ── Localization head ───────────────────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation decoder ────────────────────────────────────────
        self.bottleneck = _double_conv(512, 1024)
        self.up5 = _UpBlock(1024, 512, 512)
        self.up4 = _UpBlock(512,  512, 256)
        self.up3 = _UpBlock(256,  256, 128)
        self.up2 = _UpBlock(128,  128,  64)
        self.up1 = _UpBlock(64,    64,  64)
        self.final_conv = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._load_weights(classifier_path, localizer_path, unet_path)
        self.to(self.device)

    def _load_weights(self, cls_path, loc_path, seg_path):
        # ── Classifier ─────────────────────────────────────────────────
        if os.path.exists(cls_path):
            state = torch.load(cls_path, map_location="cpu")
            state = {k.replace("model.", ""): v for k, v in state.items()}

            # Load backbone by rebuilding a temporary Sequential matching
            # the original features layout, then letting PyTorch map keys
            tmp = nn.Sequential(
                self.enc1, self.pool1,
                self.enc2, self.pool2,
                self.enc3[0], self.enc3[1], self.pool3,
                self.enc4[0], self.enc4[1], self.pool4,
                self.enc5[0], self.enc5[1], self.pool5,
            )
            feat_state = {k.replace("features.", ""): v
                          for k, v in state.items() if k.startswith("features.")}
            tmp.load_state_dict(feat_state, strict=True)

            cls_state = {k.replace("classifier.", ""): v
                         for k, v in state.items() if k.startswith("classifier.")}
            if cls_state:
                self.cls_head.load_state_dict(cls_state, strict=True)
            print(f"[multitask] Loaded classifier from {cls_path}")

        # ── Localizer ──────────────────────────────────────────────────
        if os.path.exists(loc_path):
            state = torch.load(loc_path, map_location="cpu")
            reg_state = {k.replace("reg_head.", ""): v
                         for k, v in state.items() if k.startswith("reg_head.")}
            if reg_state:
                self.reg_head.load_state_dict(reg_state, strict=True)
            print(f"[multitask] Loaded localizer from {loc_path}")

        # ── U-Net ──────────────────────────────────────────────────────
        if os.path.exists(seg_path):
            state = torch.load(seg_path, map_location="cpu")
            for attr, module in [
                ("bottleneck", self.bottleneck),
                ("up5",        self.up5),
                ("up4",        self.up4),
                ("up3",        self.up3),
                ("up2",        self.up2),
                ("up1",        self.up1),
                ("final_conv", self.final_conv),
            ]:
                sub = {k[len(attr)+1:]: v
                       for k, v in state.items() if k.startswith(attr + ".")}
                if sub:
                    module.load_state_dict(sub, strict=True)
            print(f"[multitask] Loaded unet from {seg_path}")

    def _encode(self, x):
        s1  = self.enc1(x)
        s2  = self.enc2(self.pool1(s1))
        s3  = self.enc3(self.pool2(s2))
        s4  = self.enc4(self.pool3(s3))
        s5  = self.enc5(self.pool4(s4))
        out = self.pool5(s5)
        return out, s1, s2, s3, s4, s5

    def forward(self, x):
        x = x.to(self.device)
        enc_out, s1, s2, s3, s4, s5 = self._encode(x)

        pooled     = self.avgpool(enc_out)
        flat       = torch.flatten(pooled, 1)
        cls_logits = self.cls_head(flat)
        bbox       = self.reg_head(flat)

        b        = self.bottleneck(enc_out)
        d5       = self.up5(b,  s5)
        d4       = self.up4(d5, s4)
        d3       = self.up3(d4, s3)
        d2       = self.up2(d3, s2)
        d1       = self.up1(d2, s1)
        seg_mask = self.final_conv(d1)

        return cls_logits, bbox, seg_mask
