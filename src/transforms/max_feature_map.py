import torch
from torch import nn

class MaxFeatureMap(nn.Module):

    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def forward(self, x):

        x1, x2 = torch.chunk(x, 2, dim=self.dim)
        return torch.maximum(x1, x2)