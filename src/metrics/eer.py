import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.metrics.base_metric import BaseMetric


class EER(BaseMetric):
    """Equal error rate in percent, accumulated over a complete partition."""

    requires_full_dataset = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        scores = torch.softmax(logits, dim=1)[:, 1]
        self.scores.append(scores.detach().cpu())
        self.labels.append(labels.detach().cpu().long())

    def compute(self) -> float:
        scores = torch.cat(self.scores).numpy()
        labels = torch.cat(self.labels).numpy()

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        index = np.argmin(np.abs(fnr - fpr))

        return float((fpr[index] + fnr[index]) / 2 * 100)

    def __call__(self, **batch):
        return self.compute()
