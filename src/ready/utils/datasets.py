"""
datasets
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

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
        # print(f"{type(image) = }, {image.dtype = }, {image.shape = }")

        label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)
        # print(f"{type(label) = }, {label.dtype = }, {label.shape = }")
        #         label = F.one_hot(label, 4).type(torch.float)
        #         print(label)
        #         label = label.reshape([4, 400, 640])
        #         print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class MobiousDataset(Dataset):
    """
    MobiousDataset
    """

    def __init__(self, f_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.f_dir = f_dir

        self.img_path = list(os.listdir(os.path.join(self.f_dir, "images")))
        self.labels_path = [i.replace(".jpg", ".png") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        label_path = os.path.join(self.f_dir, "masks", self.labels_path[idx])
        image = read_image(img_path).type(torch.float) #/ 255 #torch.Size([1, 3, 400, 640])
        # image = np.asarray(Image.open( img_path ).convert("RGB")) #torch.Size([1, 400, 640, 3])

        # label = read_image(label_path).type(torch.float) #/ 255
        # label = np.asarray(Image.open( label_path ).convert("RGBA"))
        # label = torch.tensor(label, dtype=torch.long).permute(2, 0, 1).to(torch.float)
        label = np.asarray(Image.open( label_path ).convert("L"))
        label = torch.from_numpy(label).to(torch.float)
        #label = torch.tensor(label, dtype=torch.long).permute(2, 0, 1).to(torch.float)

        # label =label.clone().detach()?
        # TO_TEST/TO_REMOVE
        # label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        # label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)
        # label = F.one_hot(label, 4).type(torch.float)
        # print(label)
        # label = label.reshape([4, 400, 640])
        # print(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.tensor(label)

