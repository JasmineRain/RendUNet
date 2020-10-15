import torch
import numpy as np


def dice_coeff(true, pred):
    eps = 1e-6
    num = pred.size(0)
    pred = (pred > 0.5).float()
    m1 = pred.view(num, -1)  # Flatten
    m2 = true.view(num, -1)  # Flatten
    inter = (m1 * m2).sum()

    return (2. * inter) / (m1.sum() + m2.sum() + eps)


def get_accuracy(pred, true, threshold=0.5):
    pred = pred > threshold
    true = true == torch.max(true)
    corr = torch.sum(pred == true)
    tensor_size = pred.size(0) * pred.size(1) * pred.size(2) * pred.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(pred, true, threshold=0.5):
    # Sensitivity == Recall
    pred = pred > threshold
    true = true == torch.max(true)

    TP = ((pred == 1) & (true == 1))
    FN = ((pred == 0) & (true == 1))

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(pred, true, threshold=0.5):
    pred = pred > threshold
    true = true == torch.max(true)

    TN = ((pred == 0) & (true == 0))
    FP = ((pred == 1) & (true == 0))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(pred, true, threshold=0.5):
    pred = pred > threshold
    true = true == torch.max(true)

    TP = ((pred == 1) & (true == 1))
    FP = ((pred == 1) & (true == 0))

    PR = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PR


def get_F1(SE, PR):
    # Sensitivity == Recall
    F1 = 2 * PR *SE / (SE + PR + 1e-6)

    return F1


if __name__ == "__main__":
    true = np.array([[[[0, 0, 0, 0],
                       [1, 1, 1, 1]],
                      [[0, 0, 0, 0],
                       [1, 1, 1, 1]],
                      [[0, 0, 0, 0],
                       [1, 1, 1, 1]]
                      ]])

    pred = np.array([[[[0, 0, 0, 1],
                       [1, 1, 1, 1]],
                      [[0, 0, 0, 0],
                       [1, 1, 0, 0]],
                      [[0, 0, 0, 1],
                       [1, 0, 0, 0]]
                      ]])

    true = torch.from_numpy(true)
    pred = torch.from_numpy(pred)
    print(dice_coeff(true, pred))
    print(get_accuracy(true=true, pred=pred))
    print(get_sensitivity(true=true, pred=pred))
    print(get_specificity(true=true, pred=pred))
    print(get_precision(true=true, pred=pred))
