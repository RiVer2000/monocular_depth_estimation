import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from skimage.transform import resize
# import os
# import matplotlib.pyplot as plt

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


class DataGenerator(Dataset):
    def __init__(self, list_IDs, labels, dim=(128, 128), n_channels=3, shuffle=True, pred=False):
        self.dim = dim
        self.list_IDs = list_IDs
        self.labels = labels
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.pred = pred
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        res = random.choice([True, False])  # Random horizontal flip

        if self.pred:
            X = self.__data_generation(ID, res)
            return torch.tensor(X, dtype=torch.float32).permute(2, 0, 1)  # Convert to PyTorch format [C, H, W]

        X, y = self.__data_generation(ID, res)
        X = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.list_IDs)

    def __data_generation(self, ID, horizontal_flip):
        X = preprocess_image(ID, horizontal_flip)

        if not self.pred:
            y = preprocess_depth_map(self.labels[ID], horizontal_flip)
            return X, y
        return X
# Test the functionality of the custom generator
# image_path = 'nyu_data/data/nyu2_test/00000_colors.png'  # Update with your actual file path
# depth_map_path = 'nyu_data/data/nyu2_test/00000_depth.png'  # Update with your actual file path
# # Process image and depth map
# image = preprocess_image(image_path, horizontal_flip=True)
# depth_map = preprocess_depth_map(depth_map_path, horizontal_flip=True)

# # Display results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Preprocessed Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(depth_map.squeeze(), cmap='gray')
# plt.title("Preprocessed Depth Map")
# plt.axis("off")

# plt.show()

# list_IDs = ["nyu_data/data/nyu2_test/00000_colors.png", "nyu_data/data/nyu2_test/00001_colors.png"]
# labels = {
#     "nyu_data/data/nyu2_test/00000_colors.png": "nyu_data/data/nyu2_test/00000_depth.png",
#     "nyu_data/data/nyu2_test/00001_colors.png": "nyu_data/data/nyu2_test/00001_depth.png",
# }

# dataset = DataGenerator(list_IDs, labels, pred=False)
# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for X_batch, y_batch in dataloader:
#     print("Image batch shape:", X_batch.shape)  # Expected: (B, C, H, W)
#     print("Depth batch shape:", y_batch.shape)  # Expected: (B, C, H, W)
#     break  # Display one batch only
