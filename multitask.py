import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11
from models.classification import Classifier
from models.localization import Localizer
from models.segmentation import UNetHead

import gdown

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # classifier_path = "checkpoints/classifier.pth"
        # localizer_path = "checkpoints/localizer.pth"
        # unet_path = "checkpoints/unet.pth"

        classifier_path = ""
        localizer_path = ""
        unet_path = ""

        gdown.download(id="https://drive.google.com/file/d/1hZd_Xd7SLK8k8uWuqaiGI0OPII70jeQ0/view?usp=sharing", output=classifier_path, quiet=False)
        gdown.download(id="https://drive.google.com/file/d/1EpVxYiVWljC6mNFtEqLfvhTllBOJRqLA/view?usp=sharing", output=localizer_path, quiet=False)
        gdown.download(id="https://drive.google.com/file/d/1j2ugYbi0uRMPDdX6ffn5KbxAHl3i8Fzo/view?usp=sharing", output=unet_path, quiet=False)

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