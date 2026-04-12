# Multi-Task Visual Perception Pipeline


## Repository Link

https://github.com/SumeetD001/DA6401_Assignment_2

---

## W&B Report

https://api.wandb.ai/links/sumeet01-iitmaana/knzrsj82

## Introduction

This project implements a multi-task visual perception pipeline using PyTorch on the Oxford-IIIT Pet Dataset. The model performs classification, object localization, and semantic segmentation in a single forward pass using a VGG11-based architecture.

---

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train classification:

```bash
python train.py --task classification --data_root ./data/pets --epochs 30 --lr 1e-3

```

Train localization:

```bash
python train.py --task localization --data_root ./data/pets --epochs 30 \
                --backbone_ckpt checkpoints/classifier.pth

```

Train segmentation:

```bash
python train.py --task segmentation --data_root ./data/pets --epochs 30 \
                --backbone_ckpt checkpoints/classifier.pth
```

---
