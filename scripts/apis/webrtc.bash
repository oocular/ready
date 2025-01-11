#!/bin/bash
set -Eeuxo pipefail
##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently

#export PYTHONPATH=${PYTHONPATH}:/workspace/holohub
cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc

if [[ $1 == LOCAL ]]; then
    echo $1 test
    clear && python webrtc_client.py
elif [[ $1 == PUBLIC ]]; then
    echo $1 test
    clear && python webrtc_client.py --cert-file MyCertificate.crt --key-file MyKey.key
else
    echo "not LOCAL nor PUBLIC"
fi
