#!/bin/bash
set -Eeuxo pipefail

SOURCE=$1 #replayer #v4l2
cd /workspace/volumes/ready/src/ready/apis/holoscan/ready/python

clear && python ready.py -c ready.yaml -l logger.log -df TRUE -s ${SOURCE}
