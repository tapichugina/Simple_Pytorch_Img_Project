import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms.v2 import Compose, Normalize,Resize,ToImage,ToDtype


def calculate_statistics(dataset_folder,batch_size):
    composer=Compose([ToImage(),ToDtype(torch.float32, scale=True)])
    dataset_temp=datasets.ImageFolder('dataset/train',transform=composer)
    loader_temp= DataLoader(dataset_temp, batch_size=batch_size, shuffle=True, num_workers=2)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for images, labels in loader_temp:
        # images: [B, 3, H, W]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    print("mean:", mean)
    print("std:", std)

    return {"Mean":mean,"Std":std}


