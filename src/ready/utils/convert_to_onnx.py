"""
https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.onnx
import torchvision

device = torch.device("cpu")

def export_model(model, device):
    """
    # Input to the model
    # size = (512, 512)
   
    ISSUES
    File "/usr/local/lib/python3.10/dist-packages/torch/onnx/utils.py", line 1966, in _run_symbolic_function
    torch.onnx.errors.UnsupportedOperatorError: 
    Exporting the operator 'aten::max_unpool2d' to ONNX opset version 14 is not supported. 
    Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.

    """
    #TOADD_validate_method?
    #print(model)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    # input size of data to the model [batch, channel, height, width]
    dummy_input = torch.randn(1, 1, 400, 640, requires_grad=False).to(device)
    torch_out = model(dummy_input)  #torch.Size([1, 4, 400, 640])
    print(os.getcwd())

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "weights/ADD_NAME.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
             "input": {0: "batch_size"},  # variable length axes
             "output": {0: "batch_size"},
         },
     )


def main(model_path, input_model_name, output_model_name):
    """
    IN: model_path, input_model_name
    OUT: output_model_name
    """
    models_path_input_name = model_path + "/" + input_model_name
    models_path_output_name = model_path + "/" + output_model_name

    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 15)

    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device("cpu"))
    )
    # model.load_state_dict(torch.load(models_path_input_name))

    model = model.eval().to(device)
    dump_input = torch.randn(
        (1, 3, 224, 224)
    )  # input size of data to the model [batch, channel, height, width]

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dump_input,  # model input (or a tuple for multiple inputs)
        models_path_output_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    """
    USAGE:
    mamba activate retaisVE
    python convert_to_onnx.py -mp $HOME/... -imn *.pth -mn *.onnx
    """
    # Parse args
    parser = ArgumentParser(description="Convert models to ONNX.")
    parser.add_argument(
        "-mp",
        "--model_path",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-imn",
        "--input_model_name",
        default="none",
        help=("Set input model name"),
    )
    parser.add_argument(
        "-omn",
        "--output_model_name",
        default="none",
        help=("Set output model name"),
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        input_model_name=args.input_model_name,
        output_model_name=args.output_model_name,
    )
