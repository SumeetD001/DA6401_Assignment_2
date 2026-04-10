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

        classifier_path = "/autograder/source/classifier.pth"
        localizer_path = "/autograder/source/localizer.pth"
        unet_path = "/autograder/source/unet.pth"

        gdown.download(id="1jLOuo5nCN6GEILJHzQhtOodT7uTrzXCo", output=classifier_path, quiet=False)
        gdown.download(id="1xf97BH1Tv2wVw0svkL413tjsVy4KnTWm", output=localizer_path, quiet=False)
        gdown.download(id="1V53JIwJwpfOsDZDziLP1hqsEKUVYlL0n", output=unet_path, quiet=False)

        self.backbone = VGG11Encoder()

        self.classifier = Classifier()
        self.localizer = Localizer()
        self.segmenter = UNetHead()

        classifier_ckpt = torch.load(classifier_path, map_location=torch.device("cpu"))
        self.classifier.load_state_dict(classifier_ckpt["state_dict"])

        localizer_ckpt = torch.load(localizer_path, map_location=torch.device("cpu"))
        self.localizer.load_state_dict(localizer_ckpt["state_dict"])
        
        segmenter_ckpt = torch.load(unet_path, map_location=torch.device("cpu"))
        self.segmenter.load_state_dict(segmenter_ckpt["state_dict"])

    def forward(self,x):
        feat = self.backbone(x)
        flat = torch.flatten(feat,1)

        cls = self.classifier(flat)
        box = self.localizer(flat)
        seg = self.segmenter(feat)
        seg = F.interpolate(seg, size=(224,224))

        return {
            "classification": cls,
            "localization": box*224,
            "segmentation": seg
        }
