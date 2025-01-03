#!/bin/bash
#set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

python src/ready/apis/convert_to_onnx_and_simplify_it.py -c configs/models/unet/config_convert_to_onnx_and_simplify_it.yaml
