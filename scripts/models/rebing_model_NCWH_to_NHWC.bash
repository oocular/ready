#!/bin/bash
#set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate


cd src/ready/apis/holoscan/utils
bash holohub-utils-dependencies.bash
cd ../../../../../


#NONEEDTOBUT CHECK BEFORE REMOVE IT>export PYTHONPATH=.

##TODO CREATE A CONFIG FILE
#MODEL_NAME=_weights_29-Mar-2025_16-23-29.pth
#MODEL_PATH=~/datasets/ready/mobious/trained_models_in_cricket/29-Mar-2025_16-23-29
MODEL_NAME=_weights_28-Mar-2025_15-25-07.pth
MODEL_PATH=~/datasets/ready/mobious/models_a10080gb/28-Mar-2025_15-25-07

python src/ready/apis/holoscan/utils/graph_surgeon.py -p ${MODEL_PATH} -m ${MODEL_NAME} -c 3 -he 400 -wi 640
