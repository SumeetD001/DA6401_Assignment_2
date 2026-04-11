"""
Object Localization Model
=========================
Encoder : VGG11 convolutional backbone (features + avgpool)
Decoder : Lightweight regression head → [cx, cy, w, h] in pixel space

Design choices
--------------
* The backbone weights are *fine-tuned* (not frozen) because:
  - The Oxford-IIIT Pet dataset is small; starting from good classification
    features and allowing small adjustments gives better localisation than
    training a fresh decoder on a frozen (classification-optimised) backbone.
  - We use a low learning-rate for the backbone during training to prevent
    catastrophic forgetting (see train.py).

* Output activation: ReLU on the last layer so all coordinates are ≥ 0
  (bounding box dimensions and centre coords are always non-negative in pixel
  space).  Alternatively one could use Sigmoid * img_size, but plain ReLU
  avoids saturating gradients for large values.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class LocalizationModel(nn.Module):
    """
    Encoder-decoder for single-object bounding-box regression.

    Parameters
    ----------
    backbone_weights : str or None
        Path to a saved VGG11 state-dict.  When provided the backbone is
        initialised with those weights; otherwise random init.
    freeze_backbone : bool
        If True, gradient flow through the backbone is disabled.
    """

    def __init__(
        self,
        backbone_weights: str = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ---- Encoder ------------------------------------------------
        _vgg = VGG11(num_classes=37)
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location="cpu")
            # Strip classifier keys so only backbone weights are loaded
            backbone_state = {
                k: v for k, v in state.items()
                if k.startswith("model.features") or k.startswith("features")
            }
            # Remap keys if saved from ClassificationModel wrapper
            remapped = {}
            for k, v in backbone_state.items():
                new_key = k.replace("model.", "")
                remapped[new_key] = v
            _vgg.load_state_dict(remapped, strict=False)

        self.backbone = _vgg.features          # Sequential of conv blocks
        self.avgpool  = _vgg.avgpool           # AdaptiveAvgPool2d(7,7)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Regression head ----------------------------------------
        # 512 * 7 * 7 = 25088 input features
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),               # light regularisation for a small head

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 4),
            nn.ReLU(inplace=True),           # ensure non-negative pixel coords
        )

        self._init_head()

    # ------------------------------------------------------------------
    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        (N, 4) tensor : [cx, cy, w, h] in pixel coordinates.
        """
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.reg_head(x)
        return x
