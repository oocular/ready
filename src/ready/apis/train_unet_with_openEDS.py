"""
Train pipeline for UNET
"""

import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from loguru import logger
from omegaconf import OmegaConf

from ready.models.unet import UNet
from ready.utils.datasets import EyeDataset
from ready.utils.utils import set_data_directory

torch.cuda.empty_cache()


def save_checkpoint(state, path):
    """
    Save checkpoint method
    """
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")


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
            f"   SANITY_CHECK no[no == 0].size(): {no[no == 0].size()}, \
                                no[no == 1].size(): {no[no == 1].size()}, \
                                no[no == 2].size(): {no[no == 2].size()}, \
                                no[no == 3].size(): {no[no == 3].size()}"
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

if __name__ == "__main__":
    """
    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    #CHECK save loss
    """
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    starttime = time.time()  # print(f'Starting training loop at {startt}')

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUG_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    if not os.path.exists(FULL_MODEL_PATH):
        os.mkdir(FULL_MODEL_PATH)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    logger.info(f"cuda_available: {cuda_available}")

    trainset = EyeDataset(
        # FULL_GITHUG_DATA_PATH+"/sample-frames/val3frames"
        FULL_DATA_PATH+"/openEDS/openEDS/test"
    )
    logger.info(f"Length of trainset: {len(trainset)}")

    batch_size = config.model_hyperparameters.batch_size
    num_workers = config.model_hyperparameters.num_workers
    learning_rate = config.model_hyperparameters.learning_rate
    run_epoch = config.model_hyperparameters.epochs


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    logger.info(f"trainloader.batch_size: {trainloader.batch_size}")

    # model = UNet(nch_in=1, nch_out=4) # for openEDS with one channel and four mask
    # input_image shape torch.Size([1, 400, 640])
    # outpu_image shape torch.Size([4, 400, 640])

    model = UNet(nch_in=3, nch_out=4)  # for openEDS with 3 channels and four mask
    # input_image shape torch.Size([3, 400, 640])
    # outpu_image shape torch.Size([4, 400, 640])
    # model.summary()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())
    # TOCHECK TESTS
    # do we need default loss? loss_fn = nn.CrossEntropyLoss()
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
    # run_epoch = 100
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

    # TOTEST weight_fn
    # if weight_fn is not None:
    #     raise NotImplemented()
    # else:
    #     print(f"Starting new checkpoint. {weight_fn}")
    #     weight_fn = os.path.join(
    #         os.getcwd(),
    #         f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar",
    #     )

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print(f"Epoch {i}:")
        sum_loss = 0.0

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            # logger.info(f"{images.shape = }") #torch.Size([8, 3, 400, 640])
            # logger.info(f"{labels.shape = }") #torch.Size([8, 400, 640])

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
    current_time_stamp= datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    PATH = FULL_MODEL_PATH+"/"+datetime.now().strftime("%d-%b-%Y")
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    model_name = PATH+"/_weights_" + current_time_stamp + ".pth"
    torch.save(model.state_dict(), model_name)
    logger.info(f"Saved PyTorch Model State to {model_name}")

    # TODO
    # path_name="weights/ADD_MODEL_NAME_VAR.onnx"
    # batch_size = 1    # just a random number
    # dummy_input = torch.randn((batch_size, 1, 400, 640)).to(DEVICE)
    # export_model(model, device, path_name, dummy_input):

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")
