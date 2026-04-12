import os
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision import transforms


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224  


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


def get_seg_transform(train: bool = True):
    """Segmentation mask transform — resize only, keep integer labels."""
    return transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
    ])


class PetsDataset(Dataset):
    _BREED_MAP: dict = {}

    def __init__(
        self,
        root: str,
        split: str = "train",
        task: str = "all",
        train: bool = True,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.root  = Path(root)
        self.task  = task
        self.train = train
        self.img_tf = get_transforms(train)
        self.seg_tf = get_seg_transform(train)

        list_file = self.root / "annotations" / "list.txt"
        samples: list[dict] = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name, cls_id = parts[0], int(parts[1]) - 1 
                samples.append({"name": name, "cls_id": cls_id})

        import random
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * val_frac))
        if split == "val":
            samples = samples[:n_val]
        else:
            samples = samples[n_val:]

        self.samples = samples

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_bbox(self, name: str, orig_w: int, orig_h: int):
        """Parse Pascal VOC XML → [cx, cy, w, h] in 224-px space."""
        xml_path = self.root / "annotations" / "xmls" / f"{name}.xml"
        if not xml_path.exists():
            return torch.tensor([112.0, 112.0, 224.0, 224.0])

        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find("object")
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        sx = IMG_SIZE / orig_w
        sy = IMG_SIZE / orig_h

        xmin, xmax = xmin * sx, xmax * sx
        ymin, ymax = ymin * sy, ymax * sy

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w  = xmax - xmin
        h  = ymax - ymin
        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    # ------------------------------------------------------------------
    def _load_mask(self, name: str):
        """Load trimap mask → long tensor (values 0/1/2 after -1 offset)."""
        mask_path = self.root / "annotations" / "trimaps" / f"{name}.png"
        if not mask_path.exists():
            return torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.long)

        mask = Image.open(mask_path)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_t = torch.from_numpy(
            __import__("numpy").array(mask, dtype="int64")
        ) - 1  
        mask_t = mask_t.clamp(0, 2)
        return mask_t

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        s    = self.samples[idx]
        name = s["name"]
        cls  = s["cls_id"]

        img_path = self.root / "images" / f"{name}.jpg"
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_t = self.img_tf(img)

        if self.task == "classification":
            return img_t, torch.tensor(cls, dtype=torch.long)

        bbox = self._load_bbox(name, orig_w, orig_h)
        if self.task == "localization":
            return img_t, bbox

        mask = self._load_mask(name)
        if self.task == "segmentation":
            return img_t, mask
        return img_t, {
            "cls":  torch.tensor(cls, dtype=torch.long),
            "bbox": bbox,
            "mask": mask,
        }        return len(self.images)

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
