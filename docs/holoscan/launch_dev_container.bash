#!/bin/bash
set -Eeuxo pipefail
# USAGE
# cd $HOME/repositories/oocular/ready/docs/holoscan
# bash launch_dev_container.bash
#

cd $HOME/repositories/holohub
./holohub run-container --docker-file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile --add-volume $HOME/repositories/holoscan-sdk --add-volume $HOME/repositories/oocular/ready --add-volume $HOME/datasets
