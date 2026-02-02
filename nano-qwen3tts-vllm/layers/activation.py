import torch
from torch import nn
import torch.nn.functional as F


class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)