#!/bin/bash
set -Ee
## USAGE
## bash moving_models.bash ccxxxxx #<ADD_SERVERUSERNAME>

SERVERUSERNAME="$1"

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/files/config_model_pathfiles.yaml)

TARMODEL=weights_${model_datetimepath}_with_augmenations_${model_trainingdataname}_trained_in_${model_trainingtime}s.tar.gz

scp ${SERVERUSERNAME}@cricket.rc.ucl.ac.uk:${paths_ServerDataPath}/${model_datetimepath}/${TARMODEL} $HOME/$paths_LocalDataPath

cd $HOME/$paths_LocalDataPath
mkdir -p ${model_datetimepath}
cd ${model_datetimepath}
tar --strip-components=7 -xvzf ../${TARMODEL}
rm ${TARMODEL}

echo untar ${TARMODEL} to $HOME/$paths_LocalDataPath
