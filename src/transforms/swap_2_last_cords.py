import torch
from torch import nn

class Swap2LastCords(nn.Module):

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.transpose(-1, -2)