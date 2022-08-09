import torch
import numpy as np

from torch import nn


class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t):
        diff = x - t
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (
            abs_diff - self.C
        )
        return y.sum()


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(
            1 + torch.pow(delta_y1 / self.omega, self.alpha - y1)
        )
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)))
            * (self.alpha - y2)
            * (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)
        )
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
