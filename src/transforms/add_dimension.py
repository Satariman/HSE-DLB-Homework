import torch
from torch import nn

class AddDimension(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-3)