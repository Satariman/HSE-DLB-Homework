import torch
from torch import nn


class ExampleLoss(nn.Module):

    def __init__(self, class_weights: list[float] | None = None):
        super().__init__()

        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):

        return {"loss": self.loss(logits, labels.long())}
