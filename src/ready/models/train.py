"""
https://www.kaggle.com/code/edventy/segiris
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Function
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from segnet import SegNet
from unet import UNet

from src.ready.utils.utils import get_working_directory, set_data_directory
from src.ready.utils.utils import export_model

torch.cuda.empty_cache()
# import gc
# gc.collect()


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
        # print(img_path)
        image = read_image(img_path).type(torch.float) / 255
        # print (image)
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
        #         print(image.shape, label.shape)
        return image, label


def save_checkpoint(state, path):
    """
    Save checkpoint method
    """
    torch.save(state, path)
    print("Checkpoint saved at {}".format(path))


def norm_image(hot_img):
    """
    Normalise image
    """
    return torch.argmax(hot_img, 0)


def sanity_check(trainloader, neural_network, cuda_available):
    """
    Sanity check of trainloader
    """
    #f, axarr = plt.subplots(1, 3)

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        #print(images[0].unsqueeze(0).size()) #torch.Size([1, 1, 400, 640])
        outputs = neural_network(images[0].unsqueeze(0))
        # print("nl", labels[0], "no", outputs[0])
        print(f'   CHECK images[0].shape: {images[0].shape}, labels[0].shape: {labels[0].shape}, outputs.shape: {outputs.shape}')
        # nl = norm_image(labels[0].reshape([400, 640, 4]).
        # swapaxes(0, 2).swapaxes(1, 2)).cpu().squeeze(0)
        no = norm_image(outputs[0]).cpu().squeeze(0)
        print(f'   CHECK no[no == 0].size(): {no[no == 0].size()}, no[no == 1].size(): {no[no == 1].size()}, no[no == 2].size(): {no[no == 2].size()}, no[no == 3].size(): {no[no == 3].size()}')

        #TOSAVE_PLOTS_TEMPORALY?
        #axarr[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        #print("NLLLL", nl.shape)
        #axarr[1].imshow(labels[0].squeeze(0).cpu())
        #axarr[2].imshow(no)

        #plt.show()

        break


def main():
    """
    EyeDataset

    #TODO epoch = None
    #TODO if weight_fn is not None:
    #TODO add checkpoint
    #TODO add execution time
    #TODO save loss
    """


    starttime = time.time() #print(f'Starting training loop at {startt}')

    # print(get_working_directory())
    set_data_directory("datasets/openEDS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("weights"):
        os.mkdir("weights")
    weight_fn = None  # TO_TEST
    cuda_available = torch.cuda.is_available()
    # print(cuda_available)
    trainset = EyeDataset("openEDS/openEDS/train/")
    print("Length of trainset:", len(trainset))

    batch_size_=8 #8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")

    #model = SegNet(in_chn=1, out_chn=4, BN_momentum=0.5)
    model = UNet(nch_in=1, nch_out=4)
    #model.summary()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())
    #loss_fn = nn.CrossEntropyLoss()
    
    #TOTEST
    #class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    #REF https://github.com/say4n/pytorch-segnet/blob/master/src/train.py

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    run_epoch = 2
    epoch = None

    if weight_fn is not None:
        raise NotImplemented()
    else:
        print("Starting new checkpoint.".format(weight_fn))
        weight_fn = os.path.join(
            os.getcwd(),
            "checkpoint_{}.pth.tar".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        )

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print("Epoch {}:".format(i))
        sum_loss = 0.0

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            #print(images.shape) #torch.Size([8, 1, 400, 640])
            #print(labels.shape) #torch.Size([8, 400, 640])

            optimizer.zero_grad()
            output = model(images) #torch.Size([8, 4, 400, 640])
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if j % 100 == 0 or j == 1:  # if j % 2 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size}")
                #sanity_check(trainloader, model, cuda_available)
                save_checkpoint(
                    {
                        "epoch": run_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    "weights/o.pth",
                )

            if j == 200:
                break
        print(f"Average loss @ epoch: {sum_loss / (j*trainloader.batch_size)}")

    print("Training complete. Saving checkpoint...")
    torch.save(model.state_dict(), "weights/model.pth")

    print("Saved PyTorch Model State to model.pth")
    
    export_model(model, device)

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f'Elapsed time for the training loop: {elapsedtime} (s)')


if __name__ == "__main__":
    main()
