#!/bin/bash
#set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_PATH/../../"
 #source .venv/bin/activate #To activate the virtual environment
 source .venv/Scripts/activate #To activate the virtual environment on Windows

python src/ready/apis/train_deeplab_with_mobious.py -c configs/models/unet/config_train_deeplab_with_mobious.yaml
