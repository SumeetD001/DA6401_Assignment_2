# """
# Object Localization Model
# =========================
# Encoder : VGG11 convolutional backbone (features + avgpool)
# Decoder : Lightweight regression head → [cx, cy, w, h] in pixel space
# """

# import torch
# import torch.nn as nn
# from models.vgg11 import VGG11Encoder


# class LocalizationModel(nn.Module):
#     def __init__(self, backbone_weights=None, freeze_backbone=False):
#         super().__init__()

#         _vgg = VGG11Encoder(num_classes=37)
#         if backbone_weights is not None:
#             state = torch.load(backbone_weights, map_location="cpu")
#             # Strip ClassificationModel wrapper prefix if present
#             state = {k.replace("model.", ""): v for k, v in state.items()}
#             missing, unexpected = _vgg.load_state_dict(state, strict=False)
#             feat_loaded = [k for k in state if k.startswith("features.") and k not in missing]
#             print(f"[LocalizationModel] Loaded {len(feat_loaded)} backbone keys from {backbone_weights}")
#             if len(feat_loaded) == 0:
#                 raise RuntimeError(
#                     f"No backbone keys loaded from {backbone_weights}! "
#                     "Check checkpoint was saved from ClassificationModel or VGG11."
#                 )

#         self.backbone = _vgg.features
#         self.avgpool  = _vgg.avgpool

#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#         self.reg_head = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),

#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),

#             nn.Linear(256, 4),
#             nn.ReLU(inplace=True),
#         )

#         self._init_head()

#     def _init_head(self):
#         for m in self.reg_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.reg_head(x)        If True, gradient flow through the backbone is disabled.
#     """

#     def __init__(
#         self,
#         backbone_weights: str = None,
#         freeze_backbone: bool = False,
#     ):
#         super().__init__()

#         # ---- Encoder ------------------------------------------------
#         _vgg = VGG11(num_classes=37)
#         if backbone_weights is not None:
#             state = torch.load(backbone_weights, map_location="cpu")
#             # Strip classifier keys so only backbone weights are loaded
#             backbone_state = {
#                 k: v for k, v in state.items()
#                 if k.startswith("model.features") or k.startswith("features")
#             }
#             # Remap keys if saved from ClassificationModel wrapper
#             remapped = {}
#             for k, v in backbone_state.items():
#                 new_key = k.replace("model.", "")
#                 remapped[new_key] = v
#             _vgg.load_state_dict(remapped, strict=False)

#         self.backbone = _vgg.features          # Sequential of conv blocks
#         self.avgpool  = _vgg.avgpool           # AdaptiveAvgPool2d(7,7)

#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         # ---- Regression head ----------------------------------------
#         # 512 * 7 * 7 = 25088 input features
#         self.reg_head = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),               # light regularisation for a small head

#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),

#             nn.Linear(256, 4),
#             nn.ReLU(inplace=True),           # ensure non-negative pixel coords
#         )

#         self._init_head()

#     # ------------------------------------------------------------------
#     def _init_head(self):
#         for m in self.reg_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     # ------------------------------------------------------------------
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Returns
#         -------
#         (N, 4) tensor : [cx, cy, w, h] in pixel coordinates.
#         """
#         x = self.backbone(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.reg_head(x)
#         return x


"""
Object Localization Model
=========================
Encoder : VGG11 convolutional backbone (features + avgpool)
Decoder : Lightweight regression head → [cx, cy, w, h] in pixel space
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


class LocalizationModel(nn.Module):
    def __init__(self, backbone_weights=None, freeze_backbone=False):
        super().__init__()

        _vgg = VGG11(num_classes=37)
        if backbone_weights is not None:
            state = torch.load(backbone_weights, map_location="cpu")
            state = {k.replace("model.", ""): v for k, v in state.items()}
            missing, unexpected = _vgg.load_state_dict(state, strict=False)
            feat_loaded = [k for k in state if k.startswith("features.") and k not in missing]
            print(f"[LocalizationModel] Loaded {len(feat_loaded)} backbone keys from {backbone_weights}")
            if len(feat_loaded) == 0:
                raise RuntimeError(
                    f"No backbone keys loaded from {backbone_weights}! "
                    "Check checkpoint was saved from ClassificationModel or VGG11."
                )

        self.backbone = _vgg.features
        self.avgpool  = _vgg.avgpool

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
            # NO activation here — coordinates can be any positive value.
            # We clamp to [0, 224] at inference only, not during training,
            # so gradients flow freely through the final linear layer.
        )

        self._init_head()

    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialise the final linear bias to the centre of a 224x224 image
        # so predictions start near a reasonable default rather than zero.
        final_linear = self.reg_head[-1]
        nn.init.zeros_(final_linear.weight)
        final_linear.bias.data.copy_(torch.tensor([112.0, 112.0, 112.0, 112.0]))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.reg_head(x)
        # Clamp to valid pixel range at inference — no-op during training
        # because we don't want to block gradients
        if not self.training:
            x = x.clamp(min=0.0, max=224.0)
        return x
