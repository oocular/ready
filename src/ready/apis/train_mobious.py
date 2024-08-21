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

#from segnet import SegNet
from src.ready.models.unet import UNet
from src.ready.utils.datasets import MobiousDataset
from src.ready.utils.utils import (export_model, get_working_directory,
                                   set_data_directory)

torch.cuda.empty_cache()
# import gc
# gc.collect()


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
    Sanity check of trainloader for openEDS
    #TODO Sanity check for RTI-eyes datasets?
    """
    # f, axarr = plt.subplots(1, 3)

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        # print(images[0].unsqueeze(0).size()) #torch.Size([1, 1, 400, 640])
        outputs = neural_network(images[0].unsqueeze(0))
        # print("nl", labels[0], "no", outputs[0])
        print(
            f"   CHECK images[0].shape: {images[0].shape}, labels[0].shape: {labels[0].shape}, outputs.shape: {outputs.shape}"
        )
        # nl = norm_image(labels[0].reshape([400, 640, 4]).
        # swapaxes(0, 2).swapaxes(1, 2)).cpu().squeeze(0)
        no = norm_image(outputs[0]).cpu().squeeze(0)
        print(
            f"   CHECK no[no == 0].size(): {no[no == 0].size()}, no[no == 1].size(): {no[no == 1].size()}, no[no == 2].size(): {no[no == 2].size()}, no[no == 3].size(): {no[no == 3].size()}"
        )

        # TOSAVE_PLOTS_TEMPORALY?
        # axarr[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        # print("NLLLL", nl.shape)
        # axarr[1].imshow(labels[0].squeeze(0).cpu())
        # axarr[2].imshow(no)

        # plt.show()

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

    #set_data_directory("datasets/openEDS")
    set_data_directory("datasets/mobious/MOBIOUS")
    #print(get_working_directory())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("weights"):
        os.mkdir("weights")
    weight_fn = None  # TO_TEST

    if weight_fn is not None:
        raise NotImplemented()
    else:
        print("Starting new checkpoint.".format(weight_fn))
        weight_fn = os.path.join(
            os.getcwd(),
            "checkpoint_{}.pth.tar".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        )


    cuda_available = torch.cuda.is_available()


    trainset = MobiousDataset("test")
    print("Length of trainset:", len(trainset))

    batch_size_ = 3
    #batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")

    ##################
    #sanity_check
    #image, label = next(iter(trainloader))
    #print(image.shape, label.shape)
    #(8, 3, 1700, 3000) for image/label data
    ################

    model = UNet(nch_in=3, nch_out=4) # for openEDS with 3 channels and four mask
    #model.summary()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())
#    # CHECK: do we need default loss? loss_fn = nn.CrossEntropyLoss()

#    # TOCHECK TESTS
#    # class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
#    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
#    # REF https://github.com/say4n/pytorch-segnet/blob/master/src/train.py
#
    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    run_epoch = 1
#    run_epoch = 10
#    #02epochs: Elapsed time for the training loop: 1.53 (s) 
#    #10epochs: Elapsed time for the training loop: 7.76 (s)

    epoch = None

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print("Epoch {}:".format(i))
        sum_loss = 0.0

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            print(images.shape) #torch.Size([8, 3, 1700, 3000])
            print(labels.shape) #torch.Size([8, 3, 1700, 3000])

            optimizer.zero_grad()
            output = model(images)
            print(output.shape) #?
#            loss = loss_fn(output, labels)
#            loss.backward()
#            optimizer.step()
#
#            sum_loss += loss.item()
#            if j % 100 == 0 or j == 1:  # if j % 2 == 0 or j == 1:
#                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size}")
#                # sanity_check(trainloader, model, cuda_available)
#                save_checkpoint(
#                    {
#                        "epoch": run_epoch,
#                        "state_dict": model.state_dict(),
#                        "optimizer": optimizer.state_dict(),
#                    },
#                    "weights/o.pth",
#                )
#
#            if j == 200:
#                break
#        print(f"Average loss @ epoch: {sum_loss / (j*trainloader.batch_size)}")
#
#    print("Training complete. Saving checkpoint...")
#    modelname = datetime.now().strftime('%d-%m-%y-%H_%M__weights.pth')
#    torch.save(model.state_dict(), modelname)
#    print(f"Saved PyTorch Model State to {modelname}")
#
#    # TOCHECK
#    # path_name="weights/ADD_MODEL_NAME_VAR.onnx"
#    # batch_size = 1    # just a random number
#    # dummy_input = torch.randn((batch_size, 1, 400, 640)).to(DEVICE)
#    # export_model(model, device, path_name, dummy_input):
#
    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")


if __name__ == "__main__":
    main()
