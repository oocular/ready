"""
convert_to_onnx
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.onnx
import torchvision

from src.ready.models.unet import UNet
from src.ready.utils.utils import export_model


def main(model_path, input_model_name):
    """
    IN: model_path, input_model_name
    OUT: output_model_name

    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") #"cuda"
    channel_n = 3  # 1

    model_name = input_model_name[:-4]
    models_path_input_name = model_path + "/" + input_model_name
    models_path_output_name = model_path + "/" + model_name + ".onnx"

    # model = SegNet(in_chn=1, out_chn=4, BN_momentum=0.5)
    model = UNet(nch_in=channel_n, nch_out=4)
    model = model.to(DEVICE)
    # model = model.to(torch.device(DEVICE))

    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device(DEVICE))
    )

    model = model.eval().to(DEVICE)

    batch_size = 1  # just a random number
    dummy_input = torch.randn((batch_size, channel_n, 400, 640)).to(DEVICE)
    # input size of data to the model [batch, channel, height, width]
    # torch_out = model(dump_input)

    export_model(model, DEVICE, models_path_output_name, dummy_input)


if __name__ == "__main__":
    """
    USAGE
    conda activate readyVE
    export PYTHONPATH=.
    python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -i model-5jul2024.pth
    """

    # Parse args
    parser = ArgumentParser(description="Convert models to ONNX.")
    parser.add_argument(
        "-p",
        "--model_path",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-i",
        "--input_model_name",
        default="none",
        help=("Set input model name"),
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        input_model_name=args.input_model_name,
    )
