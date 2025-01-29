"""
convert pytorch to onnx model
"""

import os
from argparse import ArgumentParser

import onnx
import torch
import torch.nn as nn
import torch.onnx
from loguru import logger
from onnxsim import simplify
from pathlib import Path

from src.ready.models.unetvit import UNetViT
from src.ready.utils.utils import get_default_device
from src.ready.utils.helpers import export_model


def main(input_model_name):
    """
    Convert pytorch model to onnx
    From the command line:
    python src/ready/apis/pytorch2onnx.py -i unetvit_epoch_0_3.71471.pth

    IN: input_model_name with pth extension
        The input size of data to the model is [batch, channel, height, width]
        It is definted by
        dummy_input = torch.randn(1, 1, 400, 640, requires_grad=False).to(device)

    OUT: onnx model with onnx extension

    ISSUES:
        torch.onnx.errors.UnsupportedOperatorError:
        Exporting the operator 'aten::max_unpool2d' to ONNX opset version 14 is not supported.

    TODO validate_method?
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    """
    device = get_default_device()
    logger.info(f"device : {device}")

    number_of_channels = 3

    # paths for test dataset within github repo
    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/semantic-segmentation-aerial-imagery"
    MODELS_PATH = DATASET_PATH + "/models"

    model_name = input_model_name[:-4]
    models_path_input_name = MODELS_PATH + "/" + model_name + ".pth"
    models_path_onnx = MODELS_PATH + "/" + model_name + ".onnx"
    models_path_onnx_sim = MODELS_PATH + "/" + model_name + "-sim.onnx"

    model = UNetViT(n_channels=number_of_channels, n_classes=6, bilinear=True).to(device)
    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device(device))
    )
    model = model.eval().to(device)

    batch_size = 1  # just a random number
    dummy_input = torch.randn((batch_size, number_of_channels, 512, 512)).to(device)

    export_model(model, device, models_path_onnx, dummy_input)
    logger.info(f"ONNX conversion has been scussecful to create: {models_path_onnx}")

    # Simplify ONNX model
    model_onnx = onnx.load(models_path_onnx)

    model_onnx_sim, check_model_onnx = simplify(model_onnx)
    assert check_model_onnx, "Simplified ONNX model could not be validated"
    onnx.save(model_onnx_sim, models_path_onnx_sim)
    logger.info(f"Simplified ONNX model has been saved: {models_path_onnx_sim}")
    logger.info(f"You can check model properties loading models at https://netron.app/")


if __name__ == "__main__":
    """
    USAGE
    python src/ready/apis/pytorch2onnx.py -i <model_name>.pth
    FOR EXAMPLE:
    python src/ready/apis/pytorch2onnx.py -i unetvit_epoch_0_3.71471.pth
    """

    # Parse args
    parser = ArgumentParser(description="Convert models to ONNX.")
    parser.add_argument(
        "-i",
        "--input_model_name",
        default="none",
        help=("Set model name"),
    )

    args = parser.parse_args()
    main(
        input_model_name=args.input_model_name,
    )
