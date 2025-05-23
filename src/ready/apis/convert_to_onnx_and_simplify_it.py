"""
convert_to_onnx
"""
import os
from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch
import torch.onnx
from loguru import logger
from omegaconf import OmegaConf
from onnxsim import simplify

from ready.models.unet import UNet
from ready.utils.helpers import export_model

if __name__ == "__main__":
    """
    Convert pytorch to onnx model and onnx model simplification

    python src/ready/apis/convert_to_onnx_and_simplify_it.py -c config/config.yaml
    """
    parser = ArgumentParser(description="Convert models pth to ONNX and simplify it (sim.onnx)")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
    args = parser.parse_args()

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    MODEL_PATH = config.dataset.modelsPath
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)

    input_model_name = config.model.name
    input_channel_n = config.model.inputChannelNumber
    output_channel_n = config.model.outputChannelNumber
    image_height = config.model.imageHeight
    image_width = config.model.imageWidth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEVICE = torch.device("cpu") #"cuda"

    model_name = input_model_name[:-4]
    models_path_input_name = FULL_MODEL_PATH + "/" + input_model_name
    models_path_output_name = FULL_MODEL_PATH + "/" + model_name + ".onnx"

    # model = SegNet(in_chn=1, out_chn=4, BN_momentum=0.5)
    model = UNet(nch_in=input_channel_n, nch_out=output_channel_n)
    model = model.to(device)

    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device(device))
    )

    model = model.eval().to(device)

    batch_size = 1  # just a random number
    dummy_input = torch.randn((batch_size, input_channel_n, image_height, image_width)).to(device)
    # torch_out = model(dump_input)

    export_model(model, device, models_path_output_name, dummy_input)

    #### simplyfing model
    # https://github.com/daquexian/onnx-simplifier?tab=readme-ov-file
    model_path_with_model = FULL_MODEL_PATH + "/" + model_name + ".onnx"
    model_path_with_simmodel = FULL_MODEL_PATH + "/" + model_name + "-sim.onnx"

    model = onnx.load(model_path_with_model)

    ## convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, model_path_with_simmodel)
    logger.info(f"Model {model_name} has been converted to ONNX and simplified")
