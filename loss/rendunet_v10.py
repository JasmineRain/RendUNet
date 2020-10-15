import torch.nn as nn
import torch.nn.functional as F
import torch
from models.RendPoint import sampling_features


__all__ = ["MultiRendLoss_v10"]


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


class MultiRendLoss_v10(nn.Module):
    def __init__(self):
        super(MultiRendLoss_v10, self).__init__()
        self.loss = DiceLoss()

    def forward(self, output, mask):

        coarse, stage1, stage2, stage3, stage4, stage5 = output.values()

        # coarse, stage3, stage4, stage5 = output.values()

        pred0 = F.interpolate(coarse, mask.shape[-2:], mode="bilinear", align_corners=False)
        seg_loss = F.binary_cross_entropy_with_logits(pred0, mask)

        rend1 = stage1[1]
        gt_points1 = sampling_features(mask, stage1[0], mode='nearest')
        point_loss1 = F.binary_cross_entropy_with_logits(rend1, gt_points1)
        # point_loss1 = self.loss(torch.sigmoid(rend1), gt_points1)

        rend2 = stage2[1]
        gt_points2 = sampling_features(mask, stage2[0], mode='nearest')
        point_loss2 = F.binary_cross_entropy_with_logits(rend2, gt_points2)
        # point_loss2 = self.loss(torch.sigmoid(rend2), gt_points2)

        rend3 = stage3[1]
        gt_points3 = sampling_features(mask, stage3[0], mode='nearest')
        point_loss3 = F.binary_cross_entropy_with_logits(rend3, gt_points3)
        # point_loss3 = self.loss(torch.sigmoid(rend3), gt_points3)

        rend4 = stage4[1]
        gt_points4 = sampling_features(mask, stage4[0], mode='nearest')
        point_loss4 = F.binary_cross_entropy_with_logits(rend4, gt_points4)
        # point_loss4 = self.loss(torch.sigmoid(rend4), gt_points4)

        rend5 = stage5[1]
        gt_points5 = sampling_features(mask, stage5[0], mode='nearest')
        point_loss5 = F.binary_cross_entropy_with_logits(rend5, gt_points5)
        # point_loss5 = self.loss(torch.sigmoid(rend5), gt_points5)

        # point_loss = point_loss1 + point_loss2 + point_loss3 + point_loss4 + point_loss5

        point_loss = point_loss3 + point_loss4 + point_loss5

        loss = seg_loss + point_loss

        return loss
