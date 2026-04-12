import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pets_dataset import PetsDataset
from models.classification import ClassificationModel
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel
from losses.iou_loss import IoULoss


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)

        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice  = (2 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


def train_classification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PetsDataset(args.data_root, split="train", task="classification", train=True)
    val_ds   = PetsDataset(args.data_root, split="val",   task="classification", train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = ClassificationModel(num_classes=37, dropout_p=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        scheduler.step()

        print(f"[Cls] Epoch {epoch:03d}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/classifier.pth")
            print(f"  ↳ saved classifier.pth  (val_acc={best_acc:.4f})")


# ---------------------------------------------------------------------------

def train_localization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PetsDataset(args.data_root, split="train", task="localization", train=True)
    val_ds   = PetsDataset(args.data_root, split="val",   task="localization", train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = LocalizationModel(
        backbone_weights=args.backbone_ckpt,
        freeze_backbone=False,
    ).to(device)

    backbone_params = list(model.backbone.parameters()) + list(model.avgpool.parameters())
    head_params     = list(model.reg_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_mse, total_iou_l, n = 0.0, 0.0, 0.0, 0
        for imgs, bboxes in train_dl:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            l_mse = mse_loss(preds, bboxes)
            l_iou = iou_loss(preds, bboxes)
            loss  = l_mse + l_iou
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = imgs.size(0)
            total_loss  += loss.item()  * bs
            total_mse   += l_mse.item() * bs
            total_iou_l += l_iou.item() * bs
            n += bs

        tr_loss = total_loss  / n
        tr_mse  = total_mse   / n
        tr_iou  = total_iou_l / n
        model.eval()
        vl_loss, vl_mse, vl_iou, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, bboxes in val_dl:
                imgs, bboxes = imgs.to(device), bboxes.to(device)
                preds  = model(imgs)
                l_mse  = mse_loss(preds, bboxes)
                l_iou  = iou_loss(preds, bboxes)
                loss   = l_mse + l_iou
                bs = imgs.size(0)
                vl_loss += loss.item()  * bs
                vl_mse  += l_mse.item() * bs
                vl_iou  += l_iou.item() * bs
                vn += bs

        vl_loss /= vn; vl_mse /= vn; vl_iou /= vn
        scheduler.step()

        print(f"[Loc] Epoch {epoch:03d}  "
              f"train_loss={tr_loss:.4f} (mse={tr_mse:.4f} iou={tr_iou:.4f})  "
              f"val_loss={vl_loss:.4f} (mse={vl_mse:.4f} iou={vl_iou:.4f})")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  ↳ saved localizer.pth")


# ---------------------------------------------------------------------------

def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PetsDataset(args.data_root, split="train", task="segmentation", train=True)
    val_ds   = PetsDataset(args.data_root, split="val",   task="segmentation", train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SegmentationModel(
        num_classes=3,
        backbone_weights=args.backbone_ckpt,
        freeze_backbone=False,
    ).to(device)

    backbone_params = (
        list(model.enc1.parameters()) + list(model.enc2.parameters()) +
        list(model.enc3.parameters()) + list(model.enc4.parameters()) +
        list(model.enc5.parameters())
    )
    decoder_params = (
        list(model.bottleneck.parameters()) +
        list(model.up5.parameters()) + list(model.up4.parameters()) +
        list(model.up3.parameters()) + list(model.up2.parameters()) +
        list(model.up1.parameters()) + list(model.final_conv.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": decoder_params,  "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=3)

    best_val_loss = float("inf")

    def pixel_acc(logits, targets):
        return (logits.argmax(dim=1) == targets).float().mean().item()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            n  += imgs.size(0)
        tr_loss = tl / n
        model.eval()
        vl, vacc, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                vl   += loss.item()              * imgs.size(0)
                vacc += pixel_acc(logits, masks) * imgs.size(0)
                vn   += imgs.size(0)
        vl /= vn; vacc /= vn
        scheduler.step()

        print(f"[Seg] Epoch {epoch:03d}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl:.4f}  val_acc={vacc:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/unet.pth")
            print(f"  ↳ saved unet.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",          type=str,   required=True,
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root",     type=str,   default="./data/pets")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--backbone_ckpt", type=str,   default=None,
                   help="Path to classifier.pth for backbone initialisation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    elif args.task == "localization":
        train_localization(args)
    elif args.task == "segmentation":
        train_segmentation(args)
