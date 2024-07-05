"""
utils modules
"""

import os
import torch

def get_working_directory():
    """
    Test working directory
    """

    pwd = os.getcwd()
    if os.path.split(pwd)[-1] == "openeds":
        os.chdir("../../../../")  # for segnet
    else:
        os.chdir("..")  # for tests
    get_pwd = os.path.abspath(os.getcwd())
    return get_pwd


def set_data_directory(global_data_path: str):
    """
    set_data_directory with input variable:
        global_data_path. For example: "datasets/openEDS"
    """
    os.chdir(os.path.join(get_working_directory(), global_data_path))

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
    #print(os.getcwd())

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "weights/ADD_MODEL_NAME_VAR.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=16,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
             "input": {0: "batch_size"},  # variable length axes
             "output": {0: "batch_size"},
         },
     )
  

    print(f"Saved ONNX model at {os.getcwd()}")

 
