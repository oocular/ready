#!/bin/bash
set -Eeuxo pipefail
## USAGE
## bash docs/cricket/launch_container_in_cricket.bash <ADD_USERNAME (eg. ccxxxxx)>

USERNAME="$1"
TORCH_IMAGE="/home/extra/opt/nvidia/containers/pytorch:24.04-y3.sif"

apptainer run --nv --no-home -B /home/ready:/home/$USERNAME $TORCH_IMAGE /bin/bash
