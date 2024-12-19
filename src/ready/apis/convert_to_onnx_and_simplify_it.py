"""
convert_to_onnx
"""

from argparse import ArgumentParser

import onnx
import torch
import torch.onnx
from onnxsim import simplify

from src.ready.models.unet import UNet
from src.ready.utils.helpers import export_model


def main(model_path, input_model_name):
    """
    IN: model_path, input_model_name
    OUT: output_model_name

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") #"cuda"
    channel_n = 3  # 1

    model_name = input_model_name[:-4]
    models_path_input_name = model_path + "/" + input_model_name
    models_path_output_name = model_path + "/" + model_name + ".onnx"

    # model = SegNet(in_chn=1, out_chn=4, BN_momentum=0.5)
    model = UNet(nch_in=channel_n, nch_out=4)
    model = model.to(device)
    # model = model.to(torch.device(DEVICE))

    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device(device))
    )

    model = model.eval().to(device)

    batch_size = 1  # just a random number
    dummy_input = torch.randn((batch_size, channel_n, 400, 640)).to(device)
    # input size of data to the model [batch, channel, height, width]
    # torch_out = model(dump_input)

    export_model(model, device, models_path_output_name, dummy_input)

    #### simplyfing model
    # https://github.com/daquexian/onnx-simplifier?tab=readme-ov-file
    #
    model_path_with_model = model_path + "/" + model_name + ".onnx"
    model_path_with_simmodel = model_path + "/" + model_name + "-sim.onnx"

    model = onnx.load(model_path_with_model)

    ## convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, model_path_with_simmodel)



if __name__ == "__main__":
    """
    Convert pytorch to onnx model

    export PYTHONPATH=.
    python src/ready/apis/convert_to_onnx_and_simplify_it.py -p $MODEL_PATH -m model-5jul2024.pth
    """

    # Parse args
    parser = ArgumentParser(description="Convert models to ONNX and simplify it (sim.onnx)")
    parser.add_argument(
        "-p",
        "--model_path",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-m",
        "--input_model_name",
        default="none",
        help=("Set input model name"),
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        input_model_name=args.input_model_name,
    )
