import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MixLoss"]


class MixLoss(nn.Module):

    def __init__(self, alpha=1, beta=1):
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        N = target.size(0)
        smooth = 1

        input_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice_loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_loss.sum() / N

        # bce_loss = nn.BCELoss()

        bce_loss = F.binary_cross_entropy(pred, target)

        mix_loss = self.alpha * bce_loss + self.beta * dice_loss

        return mix_loss
