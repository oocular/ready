"""
datasets
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class EyeDataset(Dataset):
    """
    EyeDataset
    """

    def __init__(self, f_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.f_dir = f_dir

        self.img_path = list(os.listdir(os.path.join(self.f_dir, "images")))
        self.labels_path = [i.replace(".png", ".npy") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        image = read_image(img_path).type(torch.float) / 255

        #TODO add if for grayscale or rgb input image
        #grayscale to rgb 
        #https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315
        #print(f'grayscale to rgb')
        image = torch.stack([image,image,image],1)
        image = torch.squeeze(image)
        #print(f"{type(image) = }, {image.dtype = }, {image.shape = }")

        label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)

        #         label = F.one_hot(label, 4).type(torch.float)
        #         print(label)
        #         label = label.reshape([4, 400, 640])
        #         print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        #print(image.shape, label.shape)
        return image, label
