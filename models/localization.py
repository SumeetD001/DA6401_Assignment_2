import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class LocalizationModel(nn.Module):
    def __init__(self, backbone_weights=None, freeze_backbone=False):
        super().__init__()

        _vgg = VGG11Encoder(num_classes=37)
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

        final_linear = self.reg_head[-1]
        nn.init.zeros_(final_linear.weight)
        final_linear.bias.data.copy_(torch.tensor([112.0, 112.0, 112.0, 112.0]))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.reg_head(x)
        if not self.training:
            x = x.clamp(min=0.0, max=224.0)
        return x
