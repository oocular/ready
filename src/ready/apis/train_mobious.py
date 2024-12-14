import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.onnx
import torchvision.transforms.v2 as transforms  # https://pytorch.org/vision/main/transforms.html
from torch import nn
from torch import optim as optim

from src.ready.models.unet import UNet
from src.ready.utils.datasets import MobiousDataset
from src.ready.utils.metrics import evaluate
from src.ready.utils.utils import (HOME_PATH, sanity_check_trainloader,
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


def main(args):
    """
    Train pipeline for UNET

    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    #CHECK save loss
    ############
    # TODO LIST
    # * setup a shared path to save models when using datafrom repo (to avoid save models in repo)
    #   Currently it is using GITHUB_DATA_PATH which are ignored by .gitingore
    # * To train model with 1700x3000
    # * Test import nvidia_smi to create model vresion control: https://stackoverflow.com/questions/59567226
    # * Create a config file to train models, indidatcing paths, and other hyperparmeters
    """
    # HOME_PATH = os.path.join(Path.home(), "Desktop/nystagmus-tracking/") #MX_LOCAL_DEVICE
    HOME_PATH = os.path.join(Path.home(), "") #CRICKET_SERVER
    GITHUB_DATA_PATH = os.path.join(HOME_PATH, "ready/data/mobious") #GITHUB
    FULL_DATA_PATH = os.path.join(HOME_PATH, "datasets/mobious/MOBIOUS") #LOCAL_DEVICE

    # MODEL_PATH = os.path.join(GITHUB_DATA_PATH, "models")
    # if not os.path.exists("models"):
    #     os.mkdir("models")

    starttime = time.time()  # print(f'Starting training loop at {startt}')
    # set_data_directory(data_path="data/mobious") #data in repo
    # set_data_directory(main_path=DATA_PATH, data_path="datasets/mobious/MOBIOUS") #SERVER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()

    weight_fn = None  # TO_TEST
    if weight_fn is not None:
        raise NotImplemented()
    else:
        print(f"Starting new checkpoint. {weight_fn}""")
        weight_fn = os.path.join(
            os.getcwd(),
            f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar",
        )

    # set transforms for training images
    transforms_img = transforms.Compose([transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.5, hue = 0),
                                          transforms.ToImage(),
                                          transforms.ToDtype(torch.float32, scale=True),
                                          # ToImage and ToDtype are replacement for ToTensor which will be depreciated soon
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                        # standardisation values taken from ImageNet

    transforms_rotations = transforms.Compose([
                                            transforms.ToImage(),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(40),
                                            ])


    # Length 5; set_data_directory("ready/data")
    # trainset = MobiousDataset(
    #    GITHUB_DATA_PATH+"/sample-frames/test640x400", transform=None, target_transform=None
    #    # GITHUB_DATA_PATH+"/sample-frames/test640x400", transform=transforms_rotations, target_transform=transforms_rotations
    #   )

    ## Length 1143;  set_data_directory("datasets/mobious/MOBIOUS")
    trainset = MobiousDataset(
        FULL_DATA_PATH+"/train", transform=None, target_transform=None
        #FULL_DATA_PATH+"/train", transform=transforms_rotations, target_transform=transforms_rotations
    )

    print("Length of trainset:", len(trainset))

    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size: {trainloader.batch_size}")

    if args.debug_print_flag:
        sanity_check_trainloader(trainloader, cuda_available)

    model = UNet(nch_in=3, nch_out=4)
    # model.summary()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # TODO: check which criterium properties to setup
    loss_fn = nn.CrossEntropyLoss()
    # ?loss_fn = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    # ?loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())

    # TODO
    # class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    # REF https://github.com/say4n/pytorch-segnet/blob/master/src/train.py

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    run_epoch = 100

    #############################################
    # LOCAL NVIDIARTXA20008GBLaptopGPU
    #
    #
    # 10epochs: Elapsed time for the training loop: 7.76 (sec) #for openEDS
    # 10epochs: Elapsed time for the training loop: 4.5 (mins) #for mobious
    # 300epochs: Eliapsed time for the training loop: 6.5 (mins) #for mobious (5length trainset)
    # Average loss @ epoch: 10.22 in local
    # 300epochs: Eliapsed time for the training loop: 1.3 (mins) #for mobious (5length trainset)
    # Average loss @ epoch: 10.23 in cricket
    # run_epoch = 100
    # Average loss @ epoch: 0.0028544804081320763
    # Saved PyTorch Model State to models/_weights_10-09-24_03-46-29.pth
    # Elapsed time for the training loop: 2.1838908473650616 (mins)
    # run_epoch = 400
    # Average loss @ epoch: 0.0006139971665106714
    # Saved PyTorch Model State to models/_weights_10-09-24_04-50-40.pth
    # Elapsed time for the training loop: 13.326771756013235 (mins)


    ##############################################
    # REMOTE A100 80GB
    #
    #
    # 10epochs:
    # Eliapsed time for the training loop: 4.8 (mins) #for mobious (1143length trainset)
    # Average loss @ epoch: 12.10 in cricket
    #
    # run_epoch = 100  # noweights
    # Average loss @ epoch: 0.001589389712471593
    # Saved PyTorch Model State to models/_weights_10-09-24_06-35-14.pth
    # Elapsed time for the training loop: 47.66647284428279 (mins)
    #
    # Epoch 20: loss no-weights
    # Average loss @ epoch: 11.027751895931218
    # Saved PyTorch Model State to weights/_weights_03-09-24_22-34.pth
    # Elapsed time for the training loop: 9.677963574727377 (mins)
    #
    # Epoch 20: loss with weights
    # Average loss @ epoch: 14.233737432039701
    # Saved PyTorch Model State to weights/_weights_03-09-24_22-58.pth
    # Elapsed time for the training loop: 9.664288135369619 (mins)
    #
    # run_epoch = 200
    # Average loss @ epoch: 9.453074308542105
    # Saved PyTorch Model State to weights/_weights_04-09-24_16-31.pth
    # Elapsed time for the training loop: 96.35676774978637 (mins)
    #
    # 001 epcohs> time: 5mins; loss:0.0668
    # 002 epochs> 10mins
    # 010 epochs> without augmentations
    #     epoch loss 0.0151
    #     training time ~50.24 mins
    # 010 epochs> with augmentations (rotations)
    #    epoch loss 0.0308
    #    training time ~50.27 mins
    # 100 epochs> without augmegmnation
    #    epoch loss:0.0016
    #    training time: 508.15 mins
    # 100 epochs> wit augmegmnation
    #    epoch loss:?
    #    training time: ?
    epoch = None

    performance = {
        "accuracy": 0.0,
        "f1": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "fbeta": 0.0,
        "miou": 0.0,
        "dice": 0.0,
    }

    loss_values = []
    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print(f"############################################")
        print(f"Train loop at epoch: {i}")
        running_loss = 0.0
        num_samples, num_batches = 0, 0
        # performance_epoch = {key: 0.0 for key in performance.keys()}

        for j, data in enumerate(trainloader, 1):
            images, labels = data

            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            # print(f"output.size() {output.size()};\
            # type(output): {type(output)};\
            # pred.type: {output.type()} ")
            # torch.Size([batch_size_, 4, 400, 640]);
            # <class 'torch.Tensor'>;
            # torch.cuda.FloatTensor

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            batch_metrics = evaluate(output, labels)

            for key, value in batch_metrics.items():
                # print(f"{key}: {value:.4f}")
                performance[key] += value * len(images) # weighted by batch size

            num_samples += len(images)
            running_loss += loss.item()

            # Log every X batches
            if j % 50 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item():.4f}")
            # TODO
            #                sanity_check(trainloader, model, cuda_available)
            #                save_checkpoint(
            #                    {
            #                        "epoch": run_epoch,
            #                        "state_dict": model.state_dict(),
            #                        "optimizer": optimizer.state_dict(),
            #                    },
            #                    "models/o.pth",
            #                )
            #
            # if j == 300:
            #     break
            # # performance[key].append(average_metric)

        epoch_loss = running_loss / num_samples
        loss_values.append(epoch_loss)
        print(f"\nEpoch loss: {epoch_loss:.4f}")

        for key in performance:
            performance[key] /= num_samples
            print(f"Average {key} @ epoch: {performance[key]:.4f}")

    print("===========================")

    print("Training complete. Saving checkpoint...")
    current_time_stamp= datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    # TODO Save files in MODAL_PATH and 
    # TODO create directory with using current_time_stamp and GPU size
    if not args.debug_print_flag:
        model_name = GITHUB_DATA_PATH+"/models/_weights_" + current_time_stamp + ".pth"
        torch.save(model.state_dict(), model_name)
        print(f"Saved PyTorch Model State to {model_name}")

        json_file = GITHUB_DATA_PATH+"/models/performance_"+current_time_stamp+".json"
        text = json.dumps(performance, indent=4)
        with open(json_file, "w") as out_file_obj:
            out_file_obj.write(text)

        loss_file = GITHUB_DATA_PATH+"/models/loss_values_"+current_time_stamp+".csv"
        with open(loss_file, "w") as out_file_obj:
            for loss in loss_values:
                out_file_obj.write(f"{loss}\n")
    else:
        print("Model saving is disabled, set debug_print_flag to False (-df 0) to save model")

    # TODO
    #    batch_size = 1    # just a random number
    #    dummy_input = torch.randn((batch_size, 1, 400, 640)).to(device)
    #    export_model(model, device, path_name, dummy_input):

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")

if __name__ == "__main__":
    """
    Script to train the Mobious model using the READY API.

    Usage:
        python src/ready/apis/train_mobious.py -df <debug_flag>

    Arguments:
        -df, --debug_print_flag: Enable or disable debug printing. Use 1 (True) to enable or 0 (False) to disable.
                                 WARNING: Enabling debug mode slows performance.

    Example:
        python src/ready/apis/train_mobious.py -df 1
    """
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument(
        "-df",
        "--debug_print_flag",
        type=lambda s: s.lower() in ["true", "t", "yes", "1"],
        default=True,
        help=(
            "Set debug flag either False or True (default). \
                WARNING: Setting this to True will slow down performance of the app!"
        ),
    )

    args = parser.parse_args()
    main(args)
