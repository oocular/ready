#!/bin/bash
set -Eeuxo pipefail
## USAGE
# bash webrtc.bash logger_name.log LOCAL #PUBLIC

##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently

cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc_ready
KEYS_PATH=/workspace/volumes/datasets/ready/webrtc

#export HOLOSCAN_LOG_LEVEL=OFF
#export HOLOSCAN_LOG_LEVEL=DEBUG
#export HOLOSCAN_LOG_LEVEL=TRACE
#export HOLOSCAN_LOG_LEVEL=INFO
#export HOLOSCAN_LOG_LEVEL=ERROR

if [[ $2 == LOCAL ]]; then
    echo $2 LOCAL network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --logger_filename ${KEYS_PATH}/$1
elif [[ $2 == PUBLIC ]]; then
    echo $2 PUBLIC network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --cert-file ${KEYS_PATH}/MyCertificate.crt --key-file ${KEYS_PATH}/MyKey.key --logger_filename ${KEYS_PATH}/$1
else
    echo "not LOCAL nor PUBLIC"
fi
