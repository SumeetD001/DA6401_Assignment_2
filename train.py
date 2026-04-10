import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.pets_dataset import PetDataset
from models.vgg11 import VGG11
from models.classification import Classifier
from models.localization import Localizer
from models.segmentation import UNetHead
from losses.iou_loss import IoULoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PetDataset("images","annotations/trimaps","annotations/xmls")
loader = DataLoader(dataset,batch_size=16,shuffle=True)

backbone = VGG11().to(DEVICE)
classifier = Classifier().to(DEVICE)
localizer = Localizer().to(DEVICE)
segmenter = UNetHead().to(DEVICE)

opt = optim.Adam(
    list(backbone.parameters()) +
    list(classifier.parameters()) +
    list(localizer.parameters()) +
    list(segmenter.parameters()),
    lr=1e-5
)

cls_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
iou_loss = IoULoss()
seg_loss = nn.CrossEntropyLoss()

for epoch in range(25):
    total = 0

    for img,label,bbox,mask in loader:
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        bbox = bbox.to(DEVICE)
        mask = mask.to(DEVICE)

        feat = backbone(img)
        flat = torch.flatten(feat,1)

        cls = classifier(flat)
        box = localizer(flat)
        seg = segmenter(feat)

        seg = torch.nn.functional.interpolate(seg, size=(224,224))

        loss = (
            cls_loss(cls,label) +
            0.01 * mse_loss(box,bbox) + 
            iou_loss(box,bbox) +
            seg_loss(seg,mask)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(),1.0)
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: {total:.4f}")

torch.save({"state_dict": classifier.state_dict()}, "classifier.pth")
torch.save({"state_dict": localizer.state_dict()}, "localizer.pth")
torch.save({"state_dict": segmenter.state_dict()}, "unet.pth")
