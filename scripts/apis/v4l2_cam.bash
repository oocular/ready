#!/bin/bash
set -Eeuxo pipefail

cd /workspace/volumes/ready/src/ready/apis/holoscan/v4l2_camera/python
clear && python v4l2_camera.py
