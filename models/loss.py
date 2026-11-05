import torch.nn as nn
import torch

def neg_pearson(prediction, target, eps=1e-8):
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = prediction.pow(2).sum(dim=1).sqrt() * target.pow(2).sum(dim=1).sqrt() + eps
    return 1.0 - numerator / denominator

class PearsonLoss(nn.Module):
    def __init__(self, max_shift):
        super().__init__()
        self.max_shift = int(max_shift) # signal shift

    def forward(self, prediction, target):
        if self.max_shift <= 0:
            return neg_pearson(prediction, target).mean()
        length = prediction.shape[1]
        losses = []
        for shift in range(-self.max_shift, self.max_shift + 1): # +- 0.33
            if shift >= 0:
                pred = prediction[:, shift:]
                tgt = target[:, : length - shift]
            else:
                pred = prediction[:, :shift]
                tgt = target[:, -shift:]
            losses.append(neg_pearson(pred, tgt))
        best = torch.stack(losses, dim=0).min(dim=0).values
        return best.mean()