import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gt):
        loss = F.cross_entropy(preds, gt)
        metrics = {
            'loss': loss.item(),
        }

        return loss, metrics

