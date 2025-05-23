import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf

from ready.models.unet import UNet
from ready.utils.datasets import MobiousDataset
from ready.utils.metrics import evaluate

# TODO
# Make sure we have a common path for models to avoid looking where the model path is!

if __name__ == "__main__":
    """
    Script to test inference of Mobious models

    Usage:
        Run this script from the root directory of the project:
        python src/ready/apis/inference_mobious.py -c config/config.yaml

    Arguments:
        -p, --confi_file: Set config filename with path. Default is none.

    Tested models
        model_name="_weights_04-09-24_16-31"
            #Epoch 200:
            #Average loss @ epoch: 9.453074308542105
            #Saved PyTorch Model State to weights/_weights_04-09-24_16-31.pth
            #Elapsed time for the training loop: 96.35676774978637 (mins)

        model_name="_weights_10-09-24_03-46-29"
            # Epoch 100:
            # Average loss @ epoch: 0.0028544804081320763
            # Saved PyTorch Model State to models/_weights_10-09-24_03-46-29.pth
            # Elapsed time for the training loop: 2.1838908473650616 (mins)
        model_name="_weights_10-09-24_04-50-40"
        run_epoch = 400
                # Average loss @ epoch: 0.0006139971665106714
                # Saved PyTorch Model State to models/_weights_10-09-24_04-50-40.pth
                # Elapsed time for the training loop: 13.326771756013235 (mins)
        model_name = "_weights_08-11-24_16-38-01"
        run_epoch = 100 #noweights
                #Average loss @ epoch: 0.001589389712471593
                #Saved PyTorch Model State to models/_weights_10-09-24_06-35-14.pth
                #Elapsed time for the training loop: 47.66647284428279 (mins)
        model_name = "_weights_10-09-24_06-35-14" #BASELINE model
        model_name = "_weights_12-12-24_10-11-36" #10 epochs without augmentations
        model_name = "_weights_12-12-24_11-05-12" #10 epochs with augmentations (rotations)

    References
        skmetrics: https://github.com/MatejVitek/SSBC/blob/master/evaluation/segmentation.py
        pixel_accuracy, mIoU:
        https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation/blob/main/semantic-segmentation-pytorch.ipynb
        https://medium.com/yodayoda/segmentation-for-creating-maps-92b8d926cf7e
    """
    parser = ArgumentParser(description="Plot inference for models pth and ONNX")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
    args = parser.parse_args()

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUB_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)

    input_model_name=config.model.input_model_name
    model_name = input_model_name[:-4]
    logger.info(f"model_name {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA is available")
        import onnxruntime

    trainset = MobiousDataset(
        str(FULL_GITHUB_DATA_PATH)+"/sample-frames/test640x400_1frame_1_1i_Ll_1" # 1 frame
        # str(FULL_GITHUB_DATA_PATH)+"/sample-frames/test640x400" # 5 frames
    )
    logger.info(f"Length of trainset: {len(trainset)}")

    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    logger.info(f"trainloader.batch_size {trainloader.batch_size}")


    checkpoint_path = FULL_MODEL_PATH + '/' + str(model_name) + ".pth"
    model = UNet(nch_in=3, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if cuda_available:
        #### ONNX model
        onnx_checkpoint_path = str(FULL_MODEL_PATH) + '/' + str(model_name) + "-sim.onnx"
        ort_session = onnxruntime.InferenceSession(
            onnx_checkpoint_path, providers=["CPUExecutionProvider"]
        )

        # UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

    ### MAIN LOOP
    f, ax = plt.subplots(7, 6)
    cuda_available = torch.cuda.is_available()
    for j, data in enumerate(trainloader, 1):
        print(j)
        images, labels = data
        if cuda_available:
            images = images.cuda()
            image = images[0].unsqueeze(0)
            labels = labels.cuda()
            label = labels[0].unsqueeze(0)
            # print(f"images.size() {images.size()}")
            # #torch.Size([batch_size_, 3, 400, 640])
            # print(f"image.size() {image.size()}")
            # #torch.Size([1, 3, 400, 640])
            # print(f"labels.size() {labels.size()}")
            # #labels.size() torch.Size([batch_size_, 400, 640])
            # #WRONGtorch.Size([batch_size_, 4, 400, 640])
            # print(f"label.size() {label.size()}")
            # #label.size() torch.Size([1, 400, 640])
            # #WRONG #torch.Size([1, 4, 400, 640])

        ##PTH model
        outputs = model(image)
        # print(f"outputs.size() {outputs.size()}") #outputs.size() torch.Size([1, 4, 400, 640])
        # print(outputs[0].size())            #torch.Size([4, 400, 640])
        # print(outputs.squeeze(0).size()) #torch.Size([4, 400, 640])/

        ##PREDICTION
        pred_softmax = F.softmax(outputs, dim=1)
        # print(f"pred.size() {pred.size()}") #pred.size() torch.Size([1, 4, 400, 640])

        ## PREDICTION argmax(softmax(x))
        pred_argmax_softmax = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        # print(pred_mask.size()) #torch.Size([1, 400, 640])
        # print(pred_mask.squeeze(0).size()) #torch.Size([400, 640])

        ## PRECTION argmax(x)
        outputs_argmax = torch.argmax(outputs[0], dim=0)
        # print(outputs_argmax.size()) #torch.Size([400, 640])

        if cuda_available:
            ##ONNX model
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
            ort_outs = torch.tensor(
                np.asarray(ort_session.run(None, ort_inputs))
            )  # print(ort_outs.size()) #torch.Size([1, 1, 4, 400, 640])
            ort_outs = ort_outs.squeeze(0).squeeze(0)  #
            # print(ort_outs.size()) #torch.Size([4, 400, 640])
            ort_outs_argmax = torch.argmax(
                ort_outs, dim=0
            )  # print(ort_outs_argmax.size())#torch.Size([400, 640])

        ## Details of input image
        ### FROM holoscan-sdk API
        # tensor_.shape=(1, 3, 400, 640)
        # tensor_.min 0.0
        # tensor_.max 0.988235354423523
        # tensor_.mean 0.2402516007423401

        # image
        print(
            f"tensor.shape={image.shape}"
        )  # tensor.shape=torch.Size([1, 3, 400, 640])
        print(f"image min {torch.min(image)}")  # image min 1.0
        print(f"image max {torch.max(image)}")  # image max 252.0
        print(f"image mean {torch.mean(image)}")  # image mean 61.907188415527344

        ### FROM holoscan-sdk API
        # unet_out tensor.shape=(1, 4, 400, 640)
        # tensor.min -26.610116958618164
        # tensor.max 18.82499885559082
        # tensor.mean -10.790840148925781

        # outputs
        print(
            f"outputs.size() {outputs.size()}"
        )  # outputs.size() torch.Size([1, 4, 400, 640])
        print(f"outputs min {torch.min(outputs)}")  # outputs min -35.6884765625
        print(f"outputs max {torch.max(outputs)}")  # outputs max 18.10123634338379
        print(f"outputs mean {torch.mean(outputs)}")  # outputs mean -4.6653242111206055

        # pred_softmax
        print(
            f"outputs.size() {pred_softmax.size()}"
        )  # outputs.size() torch.Size([1, 4, 400, 640])
        print(
            f"pred_softmax min {torch.min(pred_softmax)}"
        )  # tensor.min 3.281325626518147e-23
        print(f"pred_softmax max {torch.max(pred_softmax)}")  # tensor.max 1.0
        print(f"pred_softmax mean {torch.mean(pred_softmax)}")  # tensor.mean 0.25

        if cuda_available:
            # ort_outs
            print(
                f"ort_outs.size() {ort_outs.unsqueeze(0).size()}"
            )  # ort_outs.size() torch.Size([4, 400, 640])
            print(f"ort_outs min {torch.min(ort_outs)}")  # ort_outs min -35.69083023071289
            print(f"ort_outs max {torch.max(ort_outs)}")  # ort_outs max 18.10032081604004
            print(
                f"ort_outs mean {torch.mean(ort_outs)}"
            )  # ort_outs mean -4.665435314178467

        # #######################################
        # ##PLOTTING
        # RAW IMAGE
        ax[0, 0].imshow(image.permute(0, 2, 3, 1).squeeze(0).to(torch.long).cpu())
        ax[0, 1].imshow(
            image.permute(0, 2, 3, 1).squeeze(0)[:, :, 0].to(torch.long).cpu()
        )
        ax[0, 2].imshow(
            image.permute(0, 2, 3, 1).squeeze(0)[:, :, 1].to(torch.long).cpu()
        )
        ax[0, 3].imshow(
            image.permute(0, 2, 3, 1).squeeze(0)[:, :, 2].to(torch.long).cpu()
        )
        # ax[0,1].imshow( image.permute(0, 2, 3, 1).squeeze(0)[:,:,0].to(torch.long).cpu() )
        ax[0, 0].set_ylabel("RAW [400,640,3]")
        ax[0, 1].set_title("ch0")
        ax[0, 2].set_title("ch1")
        ax[0, 3].set_title("ch2")

        # MASKS
        # print(label.permute(0, 2, 3, 1).squeeze(0).size()) #torch.Size([400, 640, 4])
        ax[1, 0].imshow(label.squeeze(0).cpu())
        ax[1, 1].imshow(label.squeeze(0).cpu() > 0)
        ax[1, 2].imshow(label.squeeze(0).cpu() > 1)
        ax[1, 3].imshow(label.squeeze(0).cpu() > 2)
        ax[1, 4].imshow(label.squeeze(0).cpu() > 3)
        ax[1, 0].set_ylabel("label[400,640]")
        ax[1, 1].set_title("label>0")
        ax[1, 2].set_title("label>1")
        ax[1, 3].set_title("label>2")
        ax[1, 4].set_title("label>3")

        # ##CHANNEL PREDICTIONS
        ax[2, 0].imshow(pred_softmax.permute(0, 2, 3, 1).squeeze(0).detach().cpu())
        ax[2, 1].imshow(pred_softmax[:, 0, :, :].squeeze(0).detach().cpu())
        ax[2, 2].imshow(pred_softmax[:, 1, :, :].squeeze(0).detach().cpu())
        ax[2, 3].imshow(pred_softmax[:, 2, :, :].squeeze(0).detach().cpu())
        ax[2, 4].imshow(pred_softmax[:, 3, :, :].squeeze(0).detach().cpu())
        ax[2, 0].set_title("pred_softmax [400,640,4]")
        ax[2, 1].set_title("ch0")
        ax[2, 2].set_title("ch1")
        ax[2, 3].set_title("ch2")
        ax[2, 4].set_title("ch3")

        # ##PREDICTIONS
        ax[3, 0].imshow(pred_argmax_softmax.squeeze(0).cpu())
        ax[3, 1].imshow(pred_argmax_softmax.squeeze(0).cpu() > 0)
        ax[3, 2].imshow(pred_argmax_softmax.squeeze(0).cpu() > 1)
        ax[3, 3].imshow(pred_argmax_softmax.squeeze(0).cpu() > 2)
        ax[3, 4].imshow(pred_argmax_softmax.squeeze(0).cpu() > 3)
        ax[3, 0].set_title("argmax(softmax(model(image))) [400, 640]")
        ax[3, 1].set_title("p_a_s>0")
        ax[3, 2].set_title("p_a_s>1")
        ax[3, 3].set_title("p_a_s>2")
        ax[3, 4].set_title("p_a_s>3")

        # PREDICTIONS argmax()
        ax[4, 0].imshow(outputs_argmax.cpu())
        ax[4, 1].imshow(outputs_argmax.cpu() > 0)
        ax[4, 2].imshow(outputs_argmax.cpu() > 1)
        ax[4, 3].imshow(outputs_argmax.cpu() > 2)
        ax[4, 4].imshow(outputs_argmax.cpu() > 3)
        ax[4, 0].set_title("argmax(outputs[0]) [400, 640]")
        ax[4, 1].set_title("outputs_argmax>0")
        ax[4, 2].set_title("outputs_argmax>1")
        ax[4, 3].set_title("outputs_argmax>2")
        ax[4, 4].set_title("outputs_argmax>3")

        if cuda_available:
            ##ONNX PREDICTIONS
            ax[5, 0].imshow(ort_outs_argmax.cpu())
            ax[5, 1].imshow(ort_outs_argmax.cpu() > 0)
            ax[5, 2].imshow(ort_outs_argmax.cpu() > 1)
            ax[5, 3].imshow(ort_outs_argmax.cpu() > 2)
            ax[5, 4].imshow(ort_outs_argmax.cpu() > 3)
            ax[5, 0].set_title("ort_outs_argmax")
            ax[5, 1].set_title("ort_outs_argmax>0")
            ax[5, 2].set_title("ort_outs_argmax>1")
            ax[5, 3].set_title("ort_outs_argmax>2")
            ax[5, 4].set_title("ort_outs_argmax>3")

            # #Clipping input data to the valid range for
            # imshow with RGB data ([0..1] for floats or [0..255] for integers).
            # Got range [-27.269308..12.142393].
            ax[6, 0].imshow(ort_outs.permute(1, 2, 0).cpu())
            ax[6, 1].imshow(ort_outs.permute(1, 2, 0)[:, :, 0].cpu())
            ax[6, 2].imshow(ort_outs.permute(1, 2, 0)[:, :, 1].cpu())
            ax[6, 3].imshow(ort_outs.permute(1, 2, 0)[:, :, 2].cpu())
            ax[6, 4].imshow(ort_outs.permute(1, 2, 0)[:, :, 3].cpu())
            ax[6, 0].set_title("onnx [400,640,4]")
            ax[6, 1].set_title("ort_outs[:,:,0]")
            ax[6, 2].set_title("ort_outs[:,:,1]")
            ax[6, 3].set_title("ort_outs[:,:,2]")
            ax[6, 4].set_title("ort_outs[:,:,3]")

        if j == 2:
            break

    plt.show()
