"""
https://github.com/daquexian/onnx-simplifier?tab=readme-ov-file
"""
import os
from argparse import ArgumentParser

import onnx
from onnxsim import simplify


def main(model_path, input_model_name):
    """
    Symplyfing onnx model
    """
    model_name = input_model_name[:-5]
    #models_path_input_name = model_path + "/" + input_model_name
    model_path_with_model = model_path + "/" + model_name + ".onnx"
    model_path_with_simmodel = model_path + "/" + model_name + "-sim.onnx"
    model = onnx.load(model_path_with_model)
 
    ## convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, model_path_with_simmodel)

if __name__ == "__main__":
    """
    USAGE
    conda activate readyVE
    export PYTHONPATH=.
    python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -m model-5jul2024.onnx
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
        "-m",
        "--model_name",
        default="none",
        help=("Set input model name"),
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        input_model_name=args.model_name,
    ) 
