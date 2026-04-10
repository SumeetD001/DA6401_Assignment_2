import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision import transforms

class PetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, xml_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.xml_dir = xml_dir

        self.images = []
        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue

            xml_path = os.path.join(xml_dir, f.replace(".jpg", ".xml"))
            if os.path.exists(xml_path):
                self.images.append(f)

        self.classes = sorted(list(set([
            "_".join(x.split("_")[:-1]) for x in self.images
        ])))
        self.cls_map = {c:i for i,c in enumerate(self.classes)}

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = Image.open(os.path.join(self.img_dir,name)).convert("RGB")

        orig_w, orig_h = img.size

        img = self.transform(img)

        mask = Image.open(os.path.join(self.mask_dir,name.replace(".jpg",".png")))
        mask = mask.resize((224,224), Image.NEAREST)
        mask = (np.array(mask) != 3).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)

        xml = ET.parse(os.path.join(self.xml_dir,name.replace(".jpg",".xml")))
        box = xml.getroot().find("object").find("bndbox")

        xmin,ymin,xmax,ymax = map(lambda x:int(box.find(x).text),
                                ["xmin","ymin","xmax","ymax"])

        scale_x = 224 / orig_w
        scale_y = 224 / orig_h

        bbox = torch.tensor([
            ((xmin+xmax)/2) * scale_x,
            ((ymin+ymax)/2) * scale_y,
            (xmax-xmin) * scale_x,
            (ymax-ymin) * scale_y
        ], dtype=torch.float32)

        label = self.cls_map["_".join(name.split("_")[:-1])]

        return img, label, bbox, mask