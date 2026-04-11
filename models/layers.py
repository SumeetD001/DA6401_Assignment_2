import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout implementation without using nn.Dropout or F.dropout.
    
    During training, each element is independently zeroed with probability p,
    and the remaining elements are scaled by 1/(1-p) to maintain expected values
    (inverted dropout). During eval, the layer is a no-op.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # Sample Bernoulli mask: keep probability = 1 - p
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
