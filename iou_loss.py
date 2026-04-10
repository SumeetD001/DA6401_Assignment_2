import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # convert (xc,yc,w,h) → (x1,y1,x2,y2)
        px1 = pred[:,0] - pred[:,2]/2
        py1 = pred[:,1] - pred[:,3]/2
        px2 = pred[:,0] + pred[:,2]/2
        py2 = pred[:,1] + pred[:,3]/2

        tx1 = target[:,0] - target[:,2]/2
        ty1 = target[:,1] - target[:,3]/2
        tx2 = target[:,0] + target[:,2]/2
        ty2 = target[:,1] + target[:,3]/2

        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_area = torch.clamp(inter_x2-inter_x1, min=0) * \
                     torch.clamp(inter_y2-inter_y1, min=0)

        pred_area = (px2-px1)*(py2-py1)
        target_area = (tx2-tx1)*(ty2-ty1)

        union = pred_area + target_area - inter_area + 1e-6
        iou = inter_area / union

        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss