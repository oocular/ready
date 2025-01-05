#!/bin/bash
set -Eeuxo pipefail

export PYTHONPATH=${PYTHONPATH}:/workspace/holohub
cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc
clear && python webrtc_client.py

