"""
utils
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

HOME_PATH = Path.home()
REPOSITORY_PATH = Path.cwd()


def set_data_directory(main_path: str = None, data_path: str = None):
    """
    set_data_directory with input variable:
        data_path.
        For example:
        set_data_directory("data/mobious/sample-frames/test640x400")
    """
    if main_path is None:
        main_path = REPOSITORY_PATH
    print(f"main_path: {main_path}")
    print(f"data_path: {data_path}")
    os.chdir(os.path.join(main_path, data_path))


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
            f"   CHECK images[0].shape: {images[0].shape}, \
                labels[0].shape: {labels[0].shape}, outputs.shape: {outputs.shape}"
        )
        # nl = norm_image(labels[0].reshape([400, 640, 4]).
        # swapaxes(0, 2).swapaxes(1, 2)).cpu().squeeze(0)
        no = norm_image(outputs[0]).cpu().squeeze(0)
        print(
            f"   CHECK no[no == 0].size(): {no[no == 0].size()}, \
                no[no == 1].size(): {no[no == 1].size()}, no[no == 2].size(): \
                    {no[no == 2].size()}, no[no == 3].size(): {no[no == 3].size()}"
        )

        # TOSAVE_PLOTS_TEMPORALY?
        #
        # axarr[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        # print("NLLLL", nl.shape)
        # axarr[1].imshow(labels[0].squeeze(0).cpu())
        # axarr[2].imshow(no)

        # plt.show()

        break


def sanity_check_trainloader(trainloader, cuda_available):
    """
    Sanity check of trainloader
    """
    # f, axarr = plt.subplots(1, 3)

    print(f"############################")
    print(f"# Sanity check of trainloader")
    print(f"# trainloader.batch_size: {trainloader.batch_size}")

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        print(f"# images.size() {images.size()};\
        type(images): {type(images)};\
        images.type: {images.type()} ")
        # images.size() torch.Size([5, 3, 400, 640])
        print(f"# labels.size() {labels.size()};\
        type(labels): {type(labels)};\
        labels.type: {labels.type()} ")
        # labels.size() torch.Size([5, 400, 640]);


        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()
            # images
            print(f"# images.size() {images.size()};\
            type(images): {type(images)};\
            images.type: {images.type()} ")
            # torch.Size([batch_size_, 3, 400, 640]);
            # <class 'torch.Tensor'>;
            # torch.cuda.FloatTensor
            # labels
            print(f"# labels.size() {labels.size()};\
            type(labels): {type(labels)};\
            labels.type: {labels.type()} ")
            # torch.Size([batch_size_, 400, 640]),
            # <class 'torch.Tensor'>, torch.cuda.LongTensor


        #TODO add sanity check for plotting image and encoded masks
        plt.subplot(2,1,1), plt.imshow(images[0].cpu().permute(1,2,0)/255), plt.colorbar()
        # plt.imshow(labels[0].squeeze().cpu().permute(1,2,0)/255), plt.colorbar()
        plt.subplot(2,1,2), plt.imshow(labels[0].cpu().squeeze(0)/255), plt.colorbar()
        plt.show()

        break

    print(f"############################")
