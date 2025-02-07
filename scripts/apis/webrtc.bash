#!/bin/bash
set -Eeuxo pipefail
## USAGE
# bash webrtc.bash logger_name.log LOCAL #PUBLIC

##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently

cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc
KEYS_PATH=/workspace/volumes/datasets/ready/webrtc

if [[ $2 == LOCAL ]]; then
    echo $2 LOCAL network
    clear && python webrtc_client.py --logger_filename $1
elif [[ $2 == PUBLIC ]]; then
    echo $2 PUBLIC network
    clear && python webrtc_client.py --cert-file ${KEYS_PATH}/MyCertificate.crt --key-file ${KEYS_PATH}/MyKey.key --logger_filename $1
else
    echo "not LOCAL nor PUBLIC"
fi
