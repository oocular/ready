"""
datasets
"""

import os

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
        self.labels_path = [i.replace(".png", ".npy") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        image = read_image(img_path).type(torch.float) / 255

        # TODO add if for grayscale or rgb input image
        # grayscale to rgb
        # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315
        # print(f'grayscale to rgb')
        image = torch.stack([image, image, image], 1)
        image = torch.squeeze(image)
        # print(f"{type(image) = }, {image.dtype = }, {image.shape = }")
        # type(image) = <class 'torch.Tensor'>, image.dtype = torch.float32,
        # image.shape = torch.Size([3, 400, 640])

        label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        # print(f"{type(label) = }, {label.dtype = }, {label.shape = }")
        label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)
        #TODO add module to DEBUG LABELS
        # import matplotlib.pyplot as plt
        # print(label.long().min(), label.long().max()) # tensor(0) tensor(3)
        # tensor(0) sclera plt.imshow(label>0)
        # tensor(1) iris  plt.imshow(label>1)
        # tensor(2) pupil plt.imshow(label>2)
        # tensor(3) background plt.imshow(label>3)

        # plt.imshow(label)
        # plt.colorbar()
        # plt.show()

        # print(f"{type(label) = }, {label.dtype = }, {label.shape = }")
        # type(label) = <class 'torch.Tensor'>, label.dtype = torch.int64,
        # label.shape = torch.Size([400, 640])

        #TODO check if this is correct
        # label = F.one_hot(label, 4).type(torch.float)
        # print(f"{type(label) = }, {label.dtype = }, {label.shape = }")
        # # type(label) = <class 'torch.Tensor'>, label.dtype = torch.float32,
        # label.shape = torch.Size([400, 640, 4])
        # label = label.reshape([4, 400, 640])
        # print(f"{type(label) = }, {label.dtype = }, {label.shape = }")
        # # type(label) = <class 'torch.Tensor'>, label.dtype = torch.float32,
        # label.shape = torch.Size([4, 400, 640])

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

        mask = Image.open(masks_path)
        # For the “P” mode, this method translates pixels through the palette.
        print(type(mask))
        print(np.array(mask).shape)

        # mask2 = mask.convert("P")
        # print(mask2)
        # print(type(mask2))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        print(type(mask))
        print(np.array(mask).shape)

        mask = mask.convert("P")
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        # print(f"************************")
        # print(mask.unique())
        # torch.set_printoptions(threshold=torch.inf)
        # print(mask)

        #TODO add sanity check for plotting encoded masks
        # plt.subplot(2,5,1), plt.imshow(mask), plt.colorbar()
        # plt.subplot(2,5,2), plt.imshow(mask>0 ), plt.colorbar()
        # plt.subplot(2,5,3), plt.imshow(mask>30), plt.colorbar()
        # plt.subplot(2,5,4), plt.imshow(mask>180), plt.colorbar()
        # plt.subplot(2,5,5), plt.imshow(mask>200), plt.colorbar()
        # # plt.show()

        encode_mask = torch.tensor(
            np.zeros((mask.shape[0], mask.shape[1])), dtype=torch.long
        )
        # 0: sclera
        encode_mask[mask > 0] = 1  # sclera (0 to 10)
        # 1: pupil
        encode_mask[mask > 30] = 2  # pupil (20 to 30)
        # # 2: iris
        encode_mask[mask > 180] = 3  # iris (40 to 180)
        # # 3: background
        # encode_mask[mask>200] = 3 #background (196 to 255)

        # print(f"************************")
        # print(encode_mask.unique())

        #TODO add sanity check for plotting encoded masks
        # plt.subplot(2,5,6), plt.imshow(encode_mask), plt.colorbar()
        # plt.subplot(2,5,7), plt.imshow(encode_mask>0 ), plt.colorbar()
        # plt.subplot(2,5,8), plt.imshow(encode_mask>1), plt.colorbar()
        # plt.subplot(2,5,9), plt.imshow(encode_mask>2), plt.colorbar()
        # plt.subplot(2,5,10), plt.imshow(encode_mask>3), plt.colorbar()
        # plt.show()

        # I tried putpalette
        # https://stackoverflow.com/questions/76131649/convert-black-and-white-image-with-color-palette-using-python
        # https://bo-li.medium.com/how-to-convert-mask-image-to-colour-map-for-mmsegmentation-52e20496192
        # https://github.com/albumentations-team/albumentations/issues/1294
        # https://stackoverflow.com/questions/67642262/python-image-types-shapes-and-channels-for-segmentation

        # label =label.clone().detach()?
        # TO_TEST/TO_REMOVE
        # label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        # label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)
        # label = F.one_hot(label, 4).type(torch.float)
        # print(label)
        # label = label.reshape([4, 400, 640])
        # print(label)

        # return image, label
        return image, encode_mask
