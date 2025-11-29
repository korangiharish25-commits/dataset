import torch
import torch.nn as nn

def dice_coeff(preds, targets, smooth=1.):
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def dice_loss(preds, targets):
    return 1.0 - dice_coeff(preds, targets)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: sigmoid outputs [B,1,H,W]
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        TP = (preds * targets).sum(dim=1)
        FP = ((1 - targets) * preds).sum(dim=1)
        FN = (targets * (1 - preds)).sum(dim=1)
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalT = (1 - Tversky) ** self.gamma
        return FocalT.mean()
