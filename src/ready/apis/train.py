"""
Train pipeline for UNET
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
from torch.autograd import Function
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# from segnet import SegNet
from src.ready.models.unet import UNet
from src.ready.utils.datasets import EyeDataset
from src.ready.utils.utils import (export_model, get_working_directory,
                                   set_data_directory)

torch.cuda.empty_cache()


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
    return torch.argmax(hot_img, dim=0)


def sanity_check(trainloader, neural_network, cuda_available):
    """
    Sanity check for trainloader and model
    """
    f, ax = plt.subplots(5, 3)

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        outputs = neural_network(images[0].unsqueeze(0))
        no = norm_image(outputs[0]).squeeze(0)
        print(
            f"   SANITY_CHECK images[0].shape: {images[0].shape}"
        )  # torch.Size([3, 400, 640])
        print(
            f"   SANITY_CHECK labels[0].shape: {labels[0].shape}"
        )  # torch.Size([400, 640])
        print(
            f"   SANITY_CHECK outputs.shape: {outputs.shape}"
        )  # torch.Size([1, 4, 400, 640])
        print(
            f"   SANITY_CHECK outputs[0].shape: {outputs[0].shape}"
        )  # torch.Size([4, 400, 640])
        print(f"   SANITY_CHECK no.shape: {no.shape}")  # torch.Size([400, 640])

        # nl = norm_image(labels[0]) #.reshape([400, 640, 4]))
        # print("NL", nl.shape)
        # swapaxes(0, 2).swapaxes(1, 2)).cpu().squeeze(0)

        print(
            f"   SANITY_CHECK no[no == 0].size(): {no[no == 0].size()}, no[no == 1].size(): {no[no == 1].size()}, no[no == 2].size(): {no[no == 2].size()}, no[no == 3].size(): {no[no == 3].size()}"
        )

        # TOSAVE_PLOTS_TEMPORALY?
        ax[0, 0].imshow(
            (images[0].permute(1, 2, 0) * 255).to(torch.long).squeeze(0).cpu()
        )
        ax[0, 1].imshow(labels[0].cpu())
        ax[1, 1].imshow(labels[0].cpu() > 0)
        ax[2, 1].imshow(labels[0].cpu() > 1)
        ax[3, 1].imshow(labels[0].cpu() > 2)
        ax[4, 1].imshow(labels[0].cpu() > 3)
        ax[0, 2].imshow(no.cpu())
        ax[1, 2].imshow(no.cpu() > 0)
        ax[2, 2].imshow(no.cpu() > 1)
        ax[3, 2].imshow(no.cpu() > 2)
        ax[4, 2].imshow(no.cpu() > 3)
        plt.show()

        break


def main():
    """
    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    #CHECK save loss
    """

    starttime = time.time()  # print(f'Starting training loop at {startt}')

    # print(get_working_directory())
    # set_data_directory("datasets/openEDS")
    set_data_directory("ready/data/openEDS")

    #####
    # TODO
    # add general path for $DATASETPATH
    # set_data_directory("datasets/DATASETPATH")
    # Split data into train, test and val (currently data is just in folders synthetic and mask-withskin)
    # RetrainUNET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("models"):
        os.mkdir("models")
    weight_fn = None  # TO_TEST
    cuda_available = torch.cuda.is_available()
    # print(cuda_available)
    # trainset = EyeDataset("openEDS/openEDS/train/")
    trainset = EyeDataset(
        "sample-frames/val3frames"
    )  # for set_data_directory("ready/data/openEDS")

    # TODO trainset = RITeye_dataset("RIT-eyes/")
    print("Length of trainset:", len(trainset))

    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")

    # model = UNet(nch_in=1, nch_out=4) # for openEDS with one channel and four mask
    # input_image shape torch.Size([1, 400, 640])
    # outpu_image shape torch.Size([4, 400, 640])

    model = UNet(nch_in=3, nch_out=4)  # for openEDS with 3 channels and four mask
    # input_image shape torch.Size([3, 400, 640])
    # outpu_image shape torch.Size([4, 400, 640])

    # model.summary()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())
    # CHECK: do we need default loss? loss_fn = nn.CrossEntropyLoss()

    # TOCHECK TESTS
    # class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    # REF https://github.com/say4n/pytorch-segnet/blob/master/src/train.py

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    #
    #
    # LOCAL NVIDIARTXA20008GBLaptopGPU
    #
    #
    # run_epoch = 1
    # run_epoch = 2
    # Epoch 2:
    # Average loss @ epoch: 0.14059916138648987
    # Elapsed time for the training loop: 0.03951407273610433 (mins)

    # run_epoch = 10
    # Epoch 10:
    # Average loss @ epoch: 0.08453492075204849
    # Elapsed time for the training loop: 0.14809249639511107 (mins)
    run_epoch = 100
    # Average loss @ epoch: 0.0025765099562704563
    # Saved PyTorch Model State to models/_weights_10-09-24_23-53-45.pth
    # Elapsed time for the training loop: 1.3849741021792095 (mins)
    #
    #
    # REMOTE A100 40GB
    #
    #
    # run_epoch = 1 #to_test

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

            # print(images.shape) #torch.Size([8, 1, 400, 640])
            # print(labels.shape) #torch.Size([8, 400, 640])

            optimizer.zero_grad()
            output = model(images)  # torch.Size([8, 4, 400, 640])
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if j % 100 == 0 or j == 1:  # if j % 2 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size}")
                # sanity_check(trainloader, model, cuda_available)
                # save_checkpoint(
                #     {
                #         "epoch": run_epoch,
                #         "state_dict": model.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #     },
                #     "models/o.pth",
                # )

            if j == 200:
                break
        print(f"Average loss @ epoch: {sum_loss / (j*trainloader.batch_size)}")

    print("Training complete. Saving checkpoint...")
    modelname = datetime.now().strftime("models/_weights_%d-%m-%y_%H-%M-%S.pth")
    torch.save(model.state_dict(), modelname)
    print(f"Saved PyTorch Model State to {modelname}")

    # TOCHECK
    # path_name="weights/ADD_MODEL_NAME_VAR.onnx"
    # batch_size = 1    # just a random number
    # dummy_input = torch.randn((batch_size, 1, 400, 640)).to(DEVICE)
    # export_model(model, device, path_name, dummy_input):

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")


if __name__ == "__main__":
    main()
