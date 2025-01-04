#!/bin/bash
set -Eeuxo pipefail

cd $HOME/repositories/holohub
./dev_container launch --add-volume $HOME/repositories/holoscan-sdk --add-volume $HOME/repositories/ready --add-volume $HOME/datasets