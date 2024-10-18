#!/bin/bash

USERNAME="$1"
cd $HOME/containers
#apptainer run --nv pytorch:24.04-y3.sif
apptainer run --nv --no-home -B /home/ready/datasets:/home/$USERNAME singularity/pytorch_24.01-y3.sif /bin/bash
