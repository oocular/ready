
import torch

def export_model(model, device, path_name, dummy_input):
    """
    # Input to the model
    # size = (512, 512)

    # input size of data to the model [batch, channel, height, width]
    #dummy_input = torch.randn(1, 1, 400, 640, requires_grad=False).to(device)
    #torch_out = model(dummy_input)  #torch.Size([1, 4, 400, 640])
    #print(os.getcwd())

    #ISSUES
    torch.onnx.errors.UnsupportedOperatorError:
    Exporting the operator 'aten::max_unpool2d' to ONNX opset version 14 is not supported.

    """
    # TOADD_validate_method?
    # print(model)
    # for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        path_name,  # where to save the model (can be a file or file-like object)
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

    print(f"Saved ONNX model: {path_name}")
