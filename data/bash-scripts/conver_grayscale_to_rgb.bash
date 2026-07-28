#!/bin/bash
set -Eeuxo pipefail

#USAGE
#bash conver_grayscale_to_rgb.bash val-000160-640wX400h

IN=$1
#OUT=$2

convert ${IN}.png -define png:color-type=2 ${IN}_rgb.png

