import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


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


class RendLoss(nn.Module):
    def __init__(self):
        super(RendLoss, self).__init__()

    def forward(self, output, mask):
        pred = torch.sigmoid(F.upsample(output['coarse'], mask.shape[-2:], mode="bilinear", align_corners=True))
        gt_points = sampling_features(mask, output['points'], mode='nearest')

        N = mask.size(0)
        smooth = 1
        input_flat = pred.view(N, -1)
        target_flat = mask.view(N, -1)
        intersection = input_flat * target_flat
        seg_loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        seg_loss = 1 - seg_loss.sum() / N

        point_loss = F.binary_cross_entropy(torch.sigmoid(output['rend']), gt_points)

        loss = seg_loss + point_loss

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
