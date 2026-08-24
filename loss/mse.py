import torch
import torch.nn.functional as F


def exp_weighted_mse(pred, target, alpha=5):
    return torch.mean(torch.exp(alpha * target) * F.mse_loss(pred, target))


def exp_weighted_mse_ts(pred, target, weights, alpha=5):
    loss = 0
    if pred.ndim == 5:
        ts = pred.shape[1]
    elif pred.ndim == 4:
        ts = pred.shape[0]

    for t in range(ts):
        loss += weights[t] * exp_weighted_mse(pred[:, t], target[:, t], alpha)

    return loss


def dual_weighted_mas(pred, target, alpha=5):
    return torch.mean(torch.max(target, pred) ** alpha * (target - pred) ** 2)


def combine_mae_mse(pred, target, alpha=5, beta1=1, beta2=1):
    return beta1 * torch.mean(torch.exp(alpha * target) * F.mse_loss(pred, target)) + beta2 * torch.mean(
        F.l1_loss(pred, target))
