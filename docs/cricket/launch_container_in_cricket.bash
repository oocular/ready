#!/bin/bash
set -Eeuxo pipefail

USERNAME="$1"
apptainer run --nv --no-home -B /home/ready:/home/$USERNAME $HOME/containers/pytorch:24.04-y3.sif /bin/bash
