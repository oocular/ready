"""
https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/

USAGE

python tests/convert_to_onnx.py --model_path $HOME/Desktop/nystagmus-tracking/weights/trained_models_in_cricket --input_model_name model_3july2024.pth --output_model_name model_3july2024_pth.onnx

ISSUES
File "/usr/local/lib/python3.10/dist-packages/torch/onnx/utils.py", line 1966, in _run_symbolic_function

File "/home/mxochicale/mambaforge/envs/readyVE/lib/python3.12/site-packages/torch/onnx/utils.py", line 1966, in _run_symbolic_function

    torch.onnx.errors.UnsupportedOperatorError: 
    Exporting the operator 'aten::max_unpool2d' to ONNX opset version 14 is not supported. 
    Please feel free to request support or submit a pull request on PyTorch GitHub: https://github.com/pytorch/pytorch/issues.

https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.onnx
import torchvision
from src.ready.openeds.segnet import SegNet


def main(model_path, input_model_name, output_model_name):
    """
    IN: model_path, input_model_name
    OUT: output_model_name

    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    #DEVICE = torch.device("cuda")

    models_path_input_name = model_path + "/" + input_model_name
    models_path_output_name = model_path + "/" + output_model_name

    model = SegNet(in_chn=1, out_chn=4, BN_momentum=0.5)
    model = model.to(DEVICE)
    #model = model.to(torch.device(DEVICE))


    #print(models_path_input_name)
    model.load_state_dict(
        #torch.load(models_path_input_name, map_location=torch.device("cpu"))
        torch.load(models_path_input_name, map_location=torch.device(DEVICE))
        #torch.load(models_path_input_name).type(torch.FloatTensor).to(DEVICE)
        #torch.load(models_path_input_name)
    )


    model = model.eval().to(DEVICE)
    #model = model.eval()


    batch_size = 1    # just a random number
    #dump_input = torch.randn((1, 1, 400, 640))  
    #dump_input = dump_input.to(DEVICE, dtype=torch.float32)

    #dump_input = torch.randn((1, 1, 400, 640), requires_grad=True)  
    dump_input = torch.randn((batch_size, 1, 400, 640)).to(DEVICE)  
    #input size of data to the model [batch, channel, height, width]

    torch_out = model(dump_input)


#    # Export the model
    print(models_path_output_name,)

    torch.onnx.export(
        model,  # model being run
        dump_input,  # model input (or a tuple for multiple inputs)
        models_path_output_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"}})


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
