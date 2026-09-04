#!/bin/bash
set -Eeuxo pipefail

cd $HOME/repositories/holohub
##holoscan-sdk-3.2.0
# ./dev_container launch --add-volume $HOME/repositories/holoscan-sdk --add-volume $HOME/repositories/oocular/ready --add-volume $HOME/datasets

##holoscan-sdk-4.6.0
./holohub run-container --no-docker-build --img holohub:local-sdk-latest --local-sdk-root $HOME/repositories/holoscan-sdk  --add-volume $HOME/repositories/holoscan-sdk --add-volume $HOME/repositories/oocular/ready --add-volume $HOME/datasets