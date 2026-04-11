"""
Multi-Task Perception Model
============================
Loads pre-trained weights from three task-specific checkpoints and wires
them into a single model with one shared VGG11 backbone.

Single forward pass yields:
  1. Classification logits  (N, 37)
  2. Bounding box           (N, 4)   — [cx, cy, w, h] in pixel space
  3. Segmentation mask      (N, 3, H, W) — class logits

Checkpoint structure expected:
  classifier.pth  — state-dict of ClassificationModel (or VGG11)
  localizer.pth   — state-dict of LocalizationModel
  unet.pth        — state-dict of SegmentationModel
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11
from models.layers import CustomDropout
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel, _double_conv, _UpBlock


# ---------------------------------------------------------------------------

class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model.

    Parameters
    ----------
    classifier_path : str   Relative path to classifier.pth
    localizer_path  : str   Relative path to localizer.pth
    unet_path       : str   Relative path to unet.pth
    num_classes     : int   Number of classification classes (default 37).
    seg_classes     : int   Number of segmentation classes (default 3).
    device          : str   "cpu" or "cuda"
    """

    def __init__(
        self,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str  = "checkpoints/localizer.pth",
        unet_path: str       = "checkpoints/unet.pth",
        num_classes: int     = 37,
        seg_classes: int     = 3,
        device: str          = "cpu",
    ):
        super().__init__()

        import gdown
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>",  output=localizer_path,  quiet=False)
        gdown.download(id="<unet.pth drive id>",       output=unet_path,       quiet=False)

        self.device = torch.device(device)

        # ── Shared backbone (VGG11 feature extractor) ──────────────────
        _base = VGG11(num_classes=num_classes)
        self.backbone = _base.features
        self.avgpool  = _base.avgpool

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

        # ── Localization regression head ────────────────────────────────
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
        # Mirror the SegmentationModel decoder (without re-building encoder)
        self.bottleneck = _double_conv(512, 1024)
        self.up5 = _UpBlock(1024, 512, 512)
        self.up4 = _UpBlock(512,  512, 256)
        self.up3 = _UpBlock(256,  256, 128)
        self.up2 = _UpBlock(128,  128, 64)
        self.up1 = _UpBlock(64,    64, 64)
        self.seg_final = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ── Load & distribute checkpoint weights ────────────────────────
        self._load_weights(classifier_path, localizer_path, unet_path)
        self.to(self.device)

    # ------------------------------------------------------------------
    def _load_weights(self, cls_path: str, loc_path: str, seg_path: str):
        map_loc = self.device

        # ---------- Classifier checkpoint ----------
        if os.path.exists(cls_path):
            cls_state = torch.load(cls_path, map_location=map_loc)
            # Support both raw VGG11 and ClassificationModel wrapper
            cls_state = {k.replace("model.", ""): v for k, v in cls_state.items()}

            backbone_state = {
                k.replace("features.", ""): v
                for k, v in cls_state.items() if k.startswith("features.")
            }
            self.backbone.load_state_dict(backbone_state, strict=False)

            cls_head_state = {
                k.replace("classifier.", ""): v
                for k, v in cls_state.items() if k.startswith("classifier.")
            }
            self.cls_head.load_state_dict(cls_head_state, strict=False)

        # ---------- Localizer checkpoint ----------
        if os.path.exists(loc_path):
            loc_state = torch.load(loc_path, map_location=map_loc)
            reg_state = {
                k.replace("reg_head.", ""): v
                for k, v in loc_state.items() if k.startswith("reg_head.")
            }
            self.reg_head.load_state_dict(reg_state, strict=False)

        # ---------- U-Net checkpoint ----------
        if os.path.exists(seg_path):
            seg_state = torch.load(seg_path, map_location=map_loc)
            for name, module in [
                ("bottleneck", self.bottleneck),
                ("up5",        self.up5),
                ("up4",        self.up4),
                ("up3",        self.up3),
                ("up2",        self.up2),
                ("up1",        self.up1),
                ("final_conv", self.seg_final),
            ]:
                sub = {
                    k[len(name) + 1:]: v
                    for k, v in seg_state.items() if k.startswith(name + ".")
                }
                if sub:
                    module.load_state_dict(sub, strict=False)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor):
        """
        Run the shared VGG11 backbone and return per-stage skip features
        needed by the segmentation decoder.
        """
        feats = self.backbone

        # Mirror the stage decomposition from SegmentationModel
        s1 = feats[0](x)
        s2 = feats[2](feats[1](s1))
        s3 = nn.Sequential(feats[4], feats[5])(feats[3](s2))
        s4 = nn.Sequential(feats[7], feats[8])(feats[6](s3))
        s5 = nn.Sequential(feats[10], feats[11])(feats[9](s4))
        out = feats[12](s5)          # after final MaxPool → 7×7

        return out, s1, s2, s3, s4, s5

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (N, 3, 224, 224) normalised input image tensor.

        Returns
        -------
        cls_logits : (N, 37)
        bbox       : (N, 4)           [cx, cy, w, h] pixel coords
        seg_mask   : (N, 3, 224, 224) class logits
        """
        x = x.to(self.device)
        enc_out, s1, s2, s3, s4, s5 = self._encode(x)

        # ── Classification branch ──
        pooled = self.avgpool(enc_out)
        flat   = torch.flatten(pooled, 1)
        cls_logits = self.cls_head(flat)

        # ── Localization branch ──
        bbox = self.reg_head(flat)

        # ── Segmentation branch ──
        b  = self.bottleneck(enc_out)
        d5 = self.up5(b,  s5)
        d4 = self.up4(d5, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        seg_mask = self.seg_final(d1)

        return cls_logits, bbox, seg_mask        self.segmenter.load_state_dict(segmenter_ckpt["state_dict"])

    def forward(self,x):
        feat = self.backbone(x)
        flat = torch.flatten(feat,1)

        cls = self.classifier(flat)
        box = self.localizer(flat)
        seg = self.segmenter(feat)
        seg = F.interpolate(seg, size=(224,224))

        return {
            "classification": cls,
            "localization": box,
            "segmentation": seg
        }
