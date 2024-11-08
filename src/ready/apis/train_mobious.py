"""
Train pipeline for UNET
"""

import os
import time
from datetime import datetime

import torch
import torch.onnx
from torch import nn
from torch import optim as optim

# from segnet import SegNet
from src.ready.models.unet import UNet
from src.ready.utils.datasets import MobiousDataset
from src.ready.utils.utils import HOME_PATH, set_data_directory

# from sklearn.metrics import jaccard_score
from src.ready.utils.metrics import evaluate # mIoU, dice
from argparse import ArgumentParser

import json

torch.cuda.empty_cache()
# import gc
# gc.collect()

# MAIN_PATH = os.path.join(HOME_PATH, "Desktop/nystagmus-tracking/") #LOCAL
MAIN_PATH = os.path.join(HOME_PATH, "") #SERVER


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
        # import matplotlib.pyplot as plt
        # axarr[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        # print("NLLLL", nl.shape)
        # axarr[1].imshow(labels[0].squeeze(0).cpu())
        # axarr[2].imshow(no)

        # plt.show()

        break


def main(args):
    """
    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    #CHECK save loss
    """
# 
    starttime = time.time()  # print(f'Starting training loop at {startt}')
    set_data_directory(data_path="data/mobious") #data in repo #change>trainset!
    # set_data_directory(main_path=MAIN_PATH, data_path="datasets/mobious/MOBIOUS") #SERVER
    # TODO train with 1700x3000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("models"):
        os.mkdir("models")
    weight_fn = None  # TO_TEST

    if weight_fn is not None:
        raise NotImplemented()
    else:
        print(f"Starting new checkpoint. {weight_fn}""")
        weight_fn = os.path.join(
            os.getcwd(),
            f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar",
        )

    cuda_available = torch.cuda.is_available()

    # Length 5; set_data_directory("ready/data")
    trainset = MobiousDataset(
       "sample-frames/test640x400"
       )

    # ## Length 1143;  set_data_directory("datasets/mobious/MOBIOUS")
    # trainset = MobiousDataset(
    #     "train"
    # )

    print("Length of trainset:", len(trainset))

    # batch_size_ = 3 #to_test
    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size: {trainloader.batch_size}")

    ##################
    # TODO create a sanity_check module
    # image, label = next(iter(trainloader))
    # print(f"image.shape: {image.shape}") #torch.Size([batch_size_, 3, 1700, 3000])
    # print(f"label.shape: {label.shape}") #torch.Size([batch_size_, 4, 1700, 3000])
    ################

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


    run_epoch = 1 #to_test
    # run_epoch = ?

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
    # REMOTE A100 40GB
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

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print("Epoch {}:".format(i))
        sum_loss = 0.0
        
        performance_epoch = {key: 0.0 for key in performance.keys()}


        for j, data in enumerate(trainloader, 1):
            images, labels = data
            # print(f"images.size() {images.size()};\
            # type(images): {type(images)};\
            # images.type: {images.type()} ")
            # images.size() torch.Size([5, 3, 400, 640])
            # print(f"labels.size() {labels.size()};\
            # type(labels): {type(labels)};\
            # labels.type: {labels.type()} ")
            # labels.size() torch.Size([5, 400, 640]);   
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()
                ## images
                # print(f"images.size() {images.size()};\
                # type(labels): {type(images)};\
                # images.type: {images.type()} ")
                # torch.Size([batch_size_, 3, 400, 640]);
                # <class 'torch.Tensor'>;
                # torch.cuda.FloatTensor
                ## labels
                # print(f"labels.size() {labels.size()};\
                # type(labels): {type(labels)};\
                # labels.type: {labels.type()} ")
                # torch.Size([batch_size_, 400, 640]),
                # <class 'torch.Tensor'>, torch.cuda.LongTensor

            optimizer.zero_grad()
            output = model(images)
            # print(f"output.size() {output.size()};\
            # type(output): {type(output)};\
            # pred.type: {output.type()} ")
            # torch.Size([batch_size_, 4, 400, 640]);
            # <class 'torch.Tensor'>;
            # torch.cuda.FloatTensor

            # labels = labels.type(torch.LongTensor).cuda()
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            batch_metrics = evaluate(output, labels)

            for key, value in batch_metrics.items():
                print(f"{key}: {value:.4f}")
                performance_epoch[key] += value

            sum_loss += loss.item()
            # Log every X batches
            if j % 50 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size:.4f}")
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
            if j == 300:
                break
            # performance[key].append(average_metric)

        average_loss = sum_loss / (j * trainloader.batch_size)

        print(f"\nAverage loss @ epoch: {average_loss:.4f}")
    for key in performance:
        performance[key] = float(performance_epoch[key] / j)
        print(f"Average {key} @ epoch: {performance[key]:.4f}")
    print("===========================")

        # print 

    print("Training complete. Saving checkpoint...")
    #TODO
    # setup a  shared path to save models when using datafrom repo (to avoid save models in repo)
    # add argument to say if we want or not save models
    if not args.debug_print_flag:
        modelname = datetime.now().strftime("models/_weights_%d-%m-%y_%H-%M-%S.pth")
        torch.save(model.state_dict(), modelname)
        print(f"Saved PyTorch Model State to {modelname}")
    else: 
        print("Model saving is disabled, set debug_print_flag to False to save model")

    # TODO
    #    batch_size = 1    # just a random number
    #    dummy_input = torch.randn((batch_size, 1, 400, 640)).to(device)
    #    export_model(model, device, path_name, dummy_input):

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")

    # export performance to json
    # reference: https://github.com/SciKit-Surgery/cmicHACKS2/blob/19d365ca92aa8f5af3da68d4c27851a1312eae31/export_eval_metrics.py
    path_to_file = datetime.now().strftime("models/performance_%d-%m-%y_%H-%M-%S.json")
    

    text = json.dumps(performance, indent=4)
    with open(path_to_file, "w") as out_file_obj:
        out_file_obj.write(text)
        
if __name__ == "__main__":
    # main()
    
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
