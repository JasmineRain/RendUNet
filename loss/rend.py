import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["RendLoss"]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        N = target.size(0)
        smooth = 1

        input_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class RendLoss(nn.Module):
    def __init__(self):
        super(RendLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, output, mask):
        pred = F.interpolate(output['coarse'], mask.shape[-2:], mode="bilinear", align_corners=False)
        gt_points = sampling_features(mask, output['points'], mode='nearest')

        # N = mask.size(0)

        # smooth = 1
        # input_flat = pred.view(N, -1)
        # target_flat = mask.view(N, -1)
        # intersection = input_flat * target_flat
        # seg_loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # seg_loss = 1 - seg_loss.sum() / N

        seg_loss = F.binary_cross_entropy_with_logits(pred, mask)

        point_loss = F.binary_cross_entropy_with_logits(output['rend'], gt_points)

        loss = seg_loss + point_loss

        return loss
