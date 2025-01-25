#!/bin/bash
set -Eeuxo pipefail
## USAGE
#bash scripts/apis/facelandmarks_rois.bash

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate

#data/bash-scripts/download_shape_predictors.bash
python src/ready/apis/facelandmarks/rois.py -c configs/apis/config_facelandmarks.yaml
