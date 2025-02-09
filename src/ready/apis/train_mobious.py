import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.onnx
import torchvision.transforms.v2 as transforms  # https://pytorch.org/vision/main/transforms.html
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch import optim as optim

from ready.models.unet import UNet
from ready.utils.datasets import MobiousDataset
from ready.utils.metrics import evaluate
from ready.utils.utils import (HOME_PATH, sanity_check_trainloader,
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
    ############
    # TODO LIST
    # * setup a shared path to save models when using datafrom repo (to avoid save models in repo)
    #   Currently it is using GITHUB_DATA_PATH which are ignored by .gitingore
    # * To train model with 1700x3000
    # * Test import nvidia_smi to create model version control: https://stackoverflow.com/questions/59567226
    # * Create a config file to train models, indidatcing paths, and other hyperparmeters
    """
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path
    debug_print_flag = config.model.debug_print_flag

    # TOCHECK
    # HOME_PATH = os.path.join(Path.home(), "") #CRICKET_SERVER

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUG_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    if not os.path.exists(FULL_MODEL_PATH):
        os.mkdir(FULL_MODEL_PATH)

    starttime = time.time()  # print(f'Starting training loop at {startt}')
    # set_data_directory(data_path="data/mobious") #data in repo
    # set_data_directory(main_path=DATA_PATH, data_path="datasets/mobious/MOBIOUS") #SERVER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()

    #TOTEST
    # weight_fn = config.model.weight_fn
    # logger.info(f"weight_fn: {weight_fn}")
    # if weight_fn is not None:
    #     raise NotImplementedError()
    # else:
    #     logger.info(f"Starting new checkpoint. {weight_fn}")
    #     weight_fn = os.path.join(
    #         os.getcwd(),
    #         f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar",
    #     )

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


    trainset = MobiousDataset(
        ## Length 5; set_data_directory("ready/data")
        # GITHUB_DATA_PATH+"/sample-frames/test640x400", transform=None, target_transform=None
        # GITHUB_DATA_PATH+"/sample-frames/test640x400", transform=transforms_rotations, target_transform=transforms_rotations
        ## Length 1143;  set_data_directory("datasets/mobious/MOBIOUS")
        FULL_DATA_PATH+"/train", transform=None, target_transform=None
        # FULL_DATA_PATH+"/train", transform=transforms_rotations, target_transform=transforms_rotations
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

    if debug_print_flag:
        sanity_check_trainloader(trainloader, cuda_available)

    model = UNet(nch_in=3, nch_out=4)
    num_params = len(torch.nn.utils.parameters_to_vector(model.parameters()))
    logger.info(f"Number of parameters in model: {num_params}")

    # model.summary()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    #    epoch loss:0.0016 (first time)/0.0014(2ndtime)
    #    training time: 508.15 mins/525.88mins
    # 100 epochs> with augmegmnation
    #    epoch loss:0.0081
    #    training time: 505.00 mins

    epoch = None
    loss_values = []
    performance = {
        "accuracy": 0.0,
        "f1": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "fbeta": 0.0,
        "miou": 0.0,
        "dice": 0.0,
    }

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        logger.info(f"#########################")
        logger.info(f"Train loop at epoch: {i}")
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

    logger.info(f"#########################")
    logger.info(f"Training complete. Saving checkpoint...")

    current_time_stamp= datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    # TODO create directory with using current_time_stamp and GPU size
    # TODO create config file to select paths and other parameters


    if not debug_print_flag:
        PATH = FULL_MODEL_PATH+"/"+datetime.now().strftime("%d-%b-%Y")
        print(PATH)
        if not os.path.exists(PATH):
            os.mkdir(PATH)

        model_name = PATH+"/_weights_" + current_time_stamp + ".pth"
        torch.save(model.state_dict(), model_name)
        logger.info(f"Saved PyTorch Model State to {model_name}")

        json_file = PATH+"/performance_"+current_time_stamp+".json"
        text = json.dumps(performance, indent=4)
        with open(json_file, "w") as out_file_obj:
            out_file_obj.write(text)

        loss_file = PATH+"/loss_values_"+current_time_stamp+".csv"
        with open(loss_file, "w") as out_file_obj:
            for loss in loss_values:
                out_file_obj.write(f"{loss}\n")
    else:
        logger.info(f"Model saving is disabled, set debug_print_flag to False (-df 0) to save model")

    # TODO
    #    batch_size = 1    # just a random number
    #    dummy_input = torch.randn((batch_size, 1, 400, 640)).to(device)
    #    export_model(model, device, path_name, dummy_input):

    endtime = time.time()
    elapsedtime = endtime - starttime
    logger.info(f"Elapsed time for the training loop: {elapsedtime} (sec)")

if __name__ == "__main__":
    """
    Script to train the Mobious model using the READY API.

    Usage:
        python src/ready/apis/train_mobious.py -df <debug_flag>

    Arguments:
        -c, --config_file: Config filename with path.
    Example:
        python src/ready/apis/train_mobious.py -c src/ready/configs/mobious_config.yaml
    """
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    main(args)
