import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11Encoder
from models.classification import Classifier
from models.localization import Localizer
from models.segmentation import UNetHead

import gdown

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        classifier_path = "classifier.pth"
        localizer_path = "localizer.pth"
        unet_path = "unet.pth"

        gdown.download(id="1hZd_Xd7SLK8k8uWuqaiGI0OPII70jeQ0", output=classifier_path, quiet=False)
        gdown.download(id="1EpVxYiVWljC6mNFtEqLfvhTllBOJRqLA", output=localizer_path, quiet=False)
        gdown.download(id="1j2ugYbi0uRMPDdX6ffn5KbxAHl3i8Fzo", output=unet_path, quiet=False)

        self.backbone = VGG11()

        self.classifier = Classifier()
        self.localizer = Localizer()
        self.segmenter = UNetHead()

        self.classifier.load_state_dict(torch.load(classifier_path)["state_dict"])
        self.localizer.load_state_dict(torch.load(localizer_path)["state_dict"])
        self.segmenter.load_state_dict(torch.load(unet_path)["state_dict"])

    def forward(self,x):
        feat = self.backbone(x)
        flat = torch.flatten(feat,1)

        cls = self.classifier(flat)
        box = self.localizer(flat)
        seg = self.segmenter(feat)
        seg = F.interpolate(seg, size=(224,224))

        return cls, box, seg
