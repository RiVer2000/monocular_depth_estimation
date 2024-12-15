import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.transform import resize

def resize_img(img, height=128):
    resized_img = resize(img, (height, int(height * 4 / 3)), preserve_range=True, mode='reflect', anti_aliasing=True)
    return resized_img

def preprocess_image(img_path, horizontal_flip=None):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if horizontal_flip:
        image = cv2.flip(image, 1)
    image = resize_img(image, height=128)
    image = np.clip(image.astype(np.float64) / 255, 0, 1)
    image = image[:, 21:149, :]
    return image

def preprocess_depth_map(depth_map_path, horizontal_flip):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if horizontal_flip:
        depth_map = cv2.flip(depth_map, 1)
    depth_map = resize_img(depth_map, height=128)
    depth_map = 1000 / np.clip(depth_map.astype(np.float64) / 255 * 1000, 0, 1000)
    depth_map = depth_map[:, 21:149]
    depth_map = np.reshape(depth_map, (128, 128, 1))
    return depth_map

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    """
    PyTorch implementation of depth loss function.
    """
    # Ensure tensors are in the correct shape
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Point-wise depth loss
    l_depth = torch.mean(torch.abs(y_pred - y_true), dim=-1)

    # Edges (using image gradients)
    dy_true, dx_true = torch.gradient(y_true, dim=(1, 2))
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(1, 2))
    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=-1)

    # Structural similarity (SSIM) index
    l_ssim = torch.clamp((1 - ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))

def depth_acc(y_true, y_pred):
    """
    Calculate depth accuracy as 1 - loss.
    """
    return 1.0 - depth_loss_function(y_true, y_pred)

def ssim(img1, img2, max_val):
    """
    Compute SSIM (Structural Similarity Index) for images.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1 * mu1
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

# testing the function
# image = preprocess_image('nyu_data/data/nyu2_test/00000_colors.png', horizontal_flip=True)
# depth_map = preprocess_depth_map('nyu_data/data/nyu2_test/00000_depth.png', horizontal_flip=True)
# y_true = torch.tensor(depth_map).permute(2, 0, 1)
# y_pred = y_true + torch.randn_like(y_true) * 0.01 
# loss = depth_loss_function(y_true, y_pred)
# acc = depth_acc(y_true, y_pred)
# print(loss, acc)

