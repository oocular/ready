#!/bin/bash
set -Eeuxo pipefail
## USAGE
# bash webrtc.bash <LOGGER_NAME.log> <NET: LOCAL/PUBLIC> <HOLOSCAN_LOG_LEVEL: OFF/DEBUG/TRACE/INFO/ERROR>

##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently

##TODO
# Move all these variables to a config file
cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc_ready
KEYS_PATH=/workspace/volumes/datasets/ready/webrtc

MODEL_PATH_MAP="/workspace/volumes/datasets/ready/mobious/models_a10080gb/15-12-24"
MODEL_NAME="_weights_15-12-24_07-00-10-sim-BHWC.onnx"

# MODEL_PATH_MAP="/workspace/volumes/datasets/ready/mobious/models_a10080gb/28-Mar-2025_15-25-07"
# MODEL_NAME="_weights_28-Mar-2025_15-25-07-sim-BHWC.onnx"

# MODEL_PATH_MAP="/workspace/volumes/datasets/ready/mobious/models_a10080gb/29-Mar-2025_16-23-29"
# MODEL_NAME="_weights_29-Mar-2025_16-23-29-sim-BHWC.onnx"

## REFERENCE
#export HOLOSCAN_LOG_LEVEL=OFF
#export HOLOSCAN_LOG_LEVEL=DEBUG
#export HOLOSCAN_LOG_LEVEL=TRACE
#export HOLOSCAN_LOG_LEVEL=INFO
#export HOLOSCAN_LOG_LEVEL=ERROR

if [[ $2 == LOCAL ]]; then
    echo $2 LOCAL network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --logger_filename ${KEYS_PATH}/$1 --model_name $MODEL_NAME --models_path_map $MODEL_PATH_MAP
elif [[ $2 == PUBLIC ]]; then
    echo $2 PUBLIC network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --cert-file ${KEYS_PATH}/MyCertificate.crt --key-file ${KEYS_PATH}/MyKey.key --logger_filename ${KEYS_PATH}/$1 --model_name $MODEL_NAME --models_path_map $MODEL_PATH_MAP
else
    echo "not LOCAL nor PUBLIC"
fi
