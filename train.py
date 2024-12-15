import torch
# import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam
# from torch.optim.lr_scheduler import LambdaLR
import os
import csv
import random
import cv2
import numpy as np
# from skimage.transform import resize
# import PIL
# from torchvision.models import resnet34
# from torchvision.models.segmentation import fcn_resnet50
import segmentation_models_pytorch as smp
import piq
# import albumentations as A  # Alternative for augmentations like `imutils`

import torch
# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import random
import csv
import os
# import matplotlib.pyplot as plt
# from PIL import Image

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    raise SystemError('GPU device not found')
print('Found GPU at:', torch.cuda.get_device_name(0))

# Constants
HEIGHT = 128
WIDTH = 128
INIT_LR = 0.0001
EPOCHS = 1
TRAIN_PATH = "/home/river2000/monocular_depth_estimation/nyu_data/data/nyu2_train.csv"
TEST_PATH = "/home/river2000/monocular_depth_estimation/nyu_data/data/nyu2_test.csv"

# Load dataset
BASE_PATH = "/home/river2000/monocular_depth_estimation/nyu_data/"

def read_csv(csv_file_path):
    with open(csv_file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        return [(os.path.join(BASE_PATH, row[0]), os.path.join(BASE_PATH, row[1])) for row in csv_reader if len(row) > 0]

def train_val_split(train_paths, val_size):
    random.shuffle(train_paths)
    len_train_paths = len(train_paths)
    i = int(len_train_paths * (1.0 - val_size))
    train = train_paths[0:i]
    val = train_paths[i:len(train_paths)]
    return train, val

def load_train_paths(train_path):
    train_paths = read_csv(train_path)
    labels = {img_path: dm_path for img_path, dm_path in train_paths}
    x_paths = [img_path for img_path, dm in train_paths]
    x_train_paths, x_val_paths = train_val_split(x_paths, 0.3)

    partition = {
        'train': x_train_paths,
        'validation': x_val_paths
    }
    return partition, labels

def load_test_paths(test_path):
    test_paths = read_csv(test_path)
    labels = {img_path: dm_path for img_path, dm_path in test_paths}
    x_paths = [img_path for img_path, dm in test_paths]

    partition = {
        'test': x_paths
    }
    return partition, labels

# Preprocessing
def normalize_img(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img

def preprocess_image(img_path, horizontal_flip=False):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    image = cv2.resize(image, (WIDTH, HEIGHT)).astype("float")
    image = normalize_img(image)

    if horizontal_flip:
        image = cv2.flip(image, 1)
    return image

def preprocess_depth_map(depth_map_path, horizontal_flip=False):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found at path: {depth_map_path}")
    depth_map = cv2.resize(depth_map, (WIDTH, HEIGHT)).astype("float")
    depth_map = normalize_img(depth_map)

    if horizontal_flip:
        depth_map = cv2.flip(depth_map, 1)

    depth_map = np.reshape(depth_map, (depth_map.shape[0], depth_map.shape[1], 1))
    return depth_map

# Dataset and DataLoader
class DepthDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=None, pred=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.transform = transform
        self.pred = pred

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        image = preprocess_image(ID)
        if self.transform:
            image = self.transform(image)
        if self.pred:
            return image
        depth_map = preprocess_depth_map(self.labels[ID])
        if self.transform:
            depth_map = self.transform(depth_map)
        return image, depth_map

# Load train and validation paths
partition, labels = load_train_paths(TRAIN_PATH)
# print(len(partition['train']), len(partition['validation']))

# Load test paths
test_partition, test_labels = load_test_paths(TEST_PATH)
# print(len(test_partition['test']))

# Create datasets and dataloaders
transform = transforms.Compose([
    transforms.ToTensor()
])

training_set = DepthDataset(partition['train'], labels, transform=transform)
training_loader = DataLoader(training_set, batch_size=16, shuffle=True)

validation_set = DepthDataset(partition['validation'], labels, transform=transform)
validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)

test_set = DepthDataset(test_partition['test'], test_labels, transform=transform)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# Model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(device)

# Loss and optimizer
import torch.nn.functional as F
import piq

def depth_loss(y_true, y_pred):
    w1, w2, w3 = 1.0, 3.0, 0.1

    l_depth = torch.mean(torch.abs(y_pred - y_true))

    # Compute gradients using finite differences
    dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dx_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    dx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    # Pad the tensors to ensure they have the same dimensions
    dy_true = F.pad(dy_true, (0, 0, 1, 0), mode='replicate')
    dx_true = F.pad(dx_true, (1, 0, 0, 0), mode='replicate')
    dy_pred = F.pad(dy_pred, (0, 0, 1, 0), mode='replicate')
    dx_pred = F.pad(dx_pred, (1, 0, 0, 0), mode='replicate')

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))

    # Normalize y_true and y_pred to the range [0, 1]
    y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

    l_ssim = torch.clip((1 - piq.ssim(y_true_norm, y_pred_norm, data_range=1.0)) * 0.5, 0, 1)

    return (w1 * l_ssim) + (w2 * l_edges) + (w3 * l_depth)

optimizer = optim.Adam(model.parameters(), lr=INIT_LR, amsgrad=True)

# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, depth_maps in training_loader:
        images, depth_maps = images.to(device).float(), depth_maps.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = depth_loss(depth_maps, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(training_loader)}")

# Evaluation
model.eval()
with torch.no_grad():
    total_loss = 0.0
    for images, depth_maps in test_loader:
        images, depth_maps = images.to(device).float(), depth_maps.to(device).float()
        outputs = model(images)
        loss = depth_loss(depth_maps, outputs)
        total_loss += loss.item()

    print(f"Test Loss: {total_loss/len(test_loader)}")

# Save model
torch.save(model.state_dict(), "./model1.pth")