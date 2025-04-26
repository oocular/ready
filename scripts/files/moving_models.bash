#!/bin/bash
set -Ee
## USAGE
## bash tarfiles.bash ccxxxxx

SERVERUSERNAME="$1"


SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/files/config_model_pathfiles.yaml)

TARMODEL=weights_${model_datetimepath}_with_augmenations_${model_trainingdataname}_trained_in_${model_trainingtime}s.tar.gz

echo "Compressed file: " $TARMODEL
scp ${SERVERUSERNAME}@cricket.rc.ucl.ac.uk:${paths_ServerDataPath}/${TARMODEL} $HOME/$paths_LocalDataPath
