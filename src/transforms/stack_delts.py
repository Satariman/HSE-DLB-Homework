import torch
import torchaudio
from torch import nn

class StackDelts(nn.Module):


    def __init__(self, delta_win_length: int = 5):
        super().__init__()

        self.delta = torchaudio.transforms.ComputeDeltas(win_length=delta_win_length)

    def forward(self, lfcc: torch.Tensor) -> torch.Tensor:
        delta = self.delta(lfcc)
        delta_delta = self.delta(delta)

        return torch.cat([lfcc, delta, delta_delta], dim=-2)