import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_bce_loss(y_true, y_pred, smooth=1e-7):
    """
    Computes Dice loss combined with binary cross-entropy loss.
    """
    # Binary cross-entropy loss
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
    
    # Dice loss
    intersection = torch.sum(y_true * y_pred)
    total = torch.sum(y_true) + torch.sum(y_pred)
    dice_loss = 1.0 - ((2 * intersection + smooth) / (total + smooth))
    
    # Combined loss
    dice_bce = bce + dice_loss
    return dice_bce


def depth_loss(y_true, y_pred, max_val=1.0, w1=1.0, w2=1.0, w3=0.1):
    """
    Computes depth loss including SSIM, edge loss, and depth loss.
    """
    # Depth loss (L1 loss)
    l_depth = torch.mean(torch.abs(y_pred - y_true), dim=(-2, -1))
    
    # Edge loss using image gradients
    dy_true, dx_true = torch.gradient(y_true, dim=(2, 3))
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(2, 3))
    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=(-2, -1))
    
    # Structural similarity (SSIM) loss
    ssim_loss = 1 - ssim(y_true, y_pred, max_val=max_val)
    l_ssim = torch.clamp(ssim_loss * 0.5, 0, 1)
    
    # Combined loss
    total_loss = (w1 * l_ssim.mean()) + (w2 * l_edges.mean()) + (w3 * l_depth.mean())
    return total_loss


def ssim(y_true, y_pred, max_val=1.0):
    """
    Computes the Structural Similarity Index (SSIM).
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_true = F.avg_pool2d(y_true, kernel_size=3, stride=1, padding=1)
    mu_pred = F.avg_pool2d(y_pred, kernel_size=3, stride=1, padding=1)

    sigma_true_sq = F.avg_pool2d(y_true * y_true, kernel_size=3, stride=1, padding=1) - mu_true ** 2
    sigma_pred_sq = F.avg_pool2d(y_pred * y_pred, kernel_size=3, stride=1, padding=1) - mu_pred ** 2
    sigma_true_pred = F.avg_pool2d(y_true * y_pred, kernel_size=3, stride=1, padding=1) - mu_true * mu_pred

    ssim_map = ((2 * mu_true * mu_pred + c1) * (2 * sigma_true_pred + c2)) / \
               ((mu_true ** 2 + mu_pred ** 2 + c1) * (sigma_true_sq + sigma_pred_sq + c2))

    return ssim_map.mean()


if __name__ == "__main__":
    # y_true = torch.rand((4, 1, 128, 128))  # Example ground truth tensor
    # y_pred = torch.rand((4, 1, 128, 128))  # Example predicted tensor
    # dice_bce = dice_bce_loss(y_true, y_pred)
    # print("Dice+BCE Loss:", dice_bce.item())
    # depth = depth_loss(y_true, y_pred)
    # print("Depth Loss:", depth.item())
    pass
