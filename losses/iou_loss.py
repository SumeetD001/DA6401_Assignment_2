import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction
        self.eps = eps

    # ------------------------------------------------------------------
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N, 4) predicted boxes  [cx, cy, w, h]
        target : (N, 4) ground-truth boxes [cx, cy, w, h]

        Returns
        -------
        Scalar loss (or per-sample tensor if reduction="none").
        Value is always in [0, 1].
        """
        pred_xyxy = self._cxcywh_to_xyxy(pred)
        tgt_xyxy = self._cxcywh_to_xyxy(target)

        inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
                    (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        tgt_area  = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0) * \
                    (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=0)

        union_area = pred_area + tgt_area - inter_area

        iou = (inter_area + self.eps) / (union_area + self.eps)

        iou = iou.clamp(0.0, 1.0)

        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, eps={self.eps}"
