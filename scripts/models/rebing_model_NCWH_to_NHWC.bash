#!/bin/bash
set -Ee
## USAGE
## bash rebing_model_NCWH_to_NHWC.bash

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../

source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/models/unet/config_convert_to_onnx_and_simplify_it.yaml)
MODEL_PATH=$HOME/${dataset_modelsPath}

source .venv/bin/activate
cd src/ready/apis/holoscan/utils
bash holohub-utils-dependencies.bash
cd ../../../../../

python src/ready/apis/holoscan/utils/graph_surgeon.py -p ${MODEL_PATH} -m ${model_name} -c ${model_inputChannelNumber} -he ${model_imageHeight} -wi ${model_imageWidth}
