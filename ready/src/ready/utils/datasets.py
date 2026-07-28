"""
datasets
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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
        #self.labels_path = [i.replace(".png", ".npy") for i in self.img_path]
        self.labels_path =  self.img_path

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        image = read_image(img_path).type(torch.float) / 255
            
        image = torch.stack([image, image, image], 1)
        image = torch.squeeze(image)

        label_path = os.path.join(self.f_dir, "labels", self.labels_path[idx])
        label = Image.open(label_path).convert("P")
        label = torch.tensor(np.array(label), dtype=torch.long)  # .unsqueeze(0)

        #label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        #label = torch.tensor(label, dtype=torch.long) 

        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = label.squeeze()

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
        self.masks_path = [i.replace(".jpg", ".png") for i in self.img_path]
        # self.labels_path = [i.replace(".jpg", ".npy") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        masks_path = os.path.join(self.f_dir, "masks", self.masks_path[idx])
        # TODO check when there is no numpy lalbels
        # labels_path = os.path.join(self.f_dir, "labels", self.labels_path[idx])
        image = read_image(img_path).type(
            torch.float
        )  # / 255 #torch.Size([1, 3, 400, 640])
        # image = np.asarray(Image.open( img_path ).convert("RGB")) #torch.Size([1, 400, 640, 3])

        # label = np.load(labels_path)
        # label = torch.tensor(label, dtype=torch.float) #.permute(2,0,1) #.unsqueeze(0)
        # print(label.size())

        ## For torch.Size([batch_size_, 400, 640]); <class 'torch.Tensor'>; torch.cuda.LongTensor`
        # mask_np = np.asarray(Image.open( masks_path ).convert("L"))
        # ?mask =  np.asarray( Image.fromarray( Image.open( masks_path )  )  )
        # ?mask = Image.fromarray(masks_path) #np.asarray(Image.open( masks_path ).convert("L"))
        # mask_t = torch.tensor(mask_np, dtype=torch.long) #uint8

        ##################
        # mask_to_class: Creating non-overlapping masks
        # https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232/3
        # This works for me
        # https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232/3
        # https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
        # https://github.com/gujingxiao/Lane-Segmentation-Solution-For-BaiduAI-Autonomous-Driving-Competition/blob/master/utils/process_labels.py
        ##################

        # # print(f"x.size() {mask.shape}; type(x): {type(mask)}; x.type: {mask.type()} ")

        mask = Image.open(masks_path).convert("P")
        # For the “P” mode, this method translates pixels through the palette.
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        encode_mask = torch.tensor(
            np.zeros((mask.shape[0], mask.shape[1])), dtype=torch.long
        )
        # 0: sclera
        encode_mask[mask > 0] = 1  # sclera (0 to 10)
        # 1: pupil
        encode_mask[mask > 30] = 2  # pupil (20 to 30)
        # 2: iris
        encode_mask[mask > 180] = 3  # iris (40 to 180)
        # 3: background
        # encode_mask[mask>200] = 3 #background (196 to 255)

        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img transform
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform:
            image = self.transform(image)

        random.seed(seed) # apply this seed to target transform
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.target_transform:
            encode_mask = self.target_transform(encode_mask)

        encode_mask=encode_mask.squeeze(0) # from torch.Size([1, 400, 640]) to #torch.Size([400, 640])

        # return image, label
        return image, encode_mask

class OPENEDS_Dataset(Dataset):
    """
    OPENEDS_Dataset
    """

    def __init__(self, f_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.f_dir = f_dir

        self.img_path = list(os.listdir(os.path.join(self.f_dir, "images")))
        self.masks_path = [i.replace(".png", ".npy") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        masks_path = os.path.join(self.f_dir, "masks", self.masks_path[idx])
        
        image = read_image(img_path).type(torch.float) / 255
        
        encode_mask = torch.tensor(np.load(masks_path), dtype=torch.long)
        
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform:
            image = self.transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        if self.target_transform:
            encode_mask = self.target_transform(encode_mask)
        
        encode_mask = encode_mask.squeeze(0)
        return image, encode_mask
    
class Rti_Eyes_Dataset(Dataset):
    """
    Rti_Eyes_Dataset
    """
    def __init__(self, f_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.f_dir = f_dir

        self.img_path = list(os.listdir(os.path.join(self.f_dir, "synthetic1")))
        self.masks_path = self.img_path  # masks have the same name as images but in a different folder

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "synthetic1", self.img_path[idx])
        masks_path = os.path.join(self.f_dir, "mask-withoutskin-noglasses1", self.masks_path[idx])

        #converting image to pytorch so the float type can be attached to it
        image = torch.tensor(np.array(Image.open(img_path).convert("RGB")), dtype=torch.float) / 255
        image = image.permute(2, 0, 1)  # convert from [H, W, 3] to [3, H, W]
        try:
            mask = np.array(Image.open(masks_path).convert("RGB"))
        except Exception as e:
            print(f"Corrupted file skipped: {masks_path} — {e}")
            image = torch.zeros(3, 128, 128)
            encode_mask = torch.zeros(128, 128, dtype=torch.long)
            return image, encode_mask
        #create 2d array of 0s, then proper pixel values for sclera, iris and pupil are initialised
        encode_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        encode_mask[np.all(mask == [0,   0,   255], axis=2)] = 1  # sclera
        encode_mask[np.all(mask == [0,   255, 0],   axis=2)] = 2  # iris
        encode_mask[np.all(mask == [255, 0,   0],   axis=2)] = 3  # pupil 
        # black pixels stay 0, the background

        encode_mask = torch.tensor(encode_mask, dtype=torch.long)

        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img transform
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform:
            image = self.transform(image)

        random.seed(seed) # apply this seed to target transform
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.target_transform:
            encode_mask = self.target_transform(encode_mask)

        encode_mask=encode_mask.squeeze(0) # from torch.Size([1, H, W]) to torch.Size([H, W])

        return image, encode_mask
