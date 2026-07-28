#!/bin/bash
set -Ee
## USAGE
# bash webrtc_ready.bash <LOGGER_NAME.log> <NET: LOCAL/PUBLIC> <HOLOSCAN_LOG_LEVEL: OFF/DEBUG/TRACE/INFO/ERROR>
#
# REFERENCE
#export HOLOSCAN_LOG_LEVEL=OFF
#export HOLOSCAN_LOG_LEVEL=DEBUG
#export HOLOSCAN_LOG_LEVEL=TRACE
#export HOLOSCAN_LOG_LEVEL=INFO
#export HOLOSCAN_LOG_LEVEL=ERROR
#
##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently


SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../

source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/apis/config_webrtc_ready.yaml)
cd ${workspace_apiPath}

if [[ $2 == LOCAL ]]; then
    echo $2 LOCAL network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --logger_filename ${workspace_keysPath}/$1 --model_name ${model_name} --models_path_map ${model_pathMap} --source $4 --enable_recording $5 --recording_directory ${recorder_directory} --recording_basename ${recorder_basename}
elif [[ $2 == PUBLIC ]]; then
    echo $2 PUBLIC network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --cert-file ${workspace_keysPath}/MyCertificate.crt --key-file ${workspace_keysPath}/MyKey.key --logger_filename ${workspace_keysPath}/$1 --model_name ${model_name} --models_path_map ${model_pathMap} --source $4 --enable_recording $5 --recording_directory ${recorder_directory} --recording_basename ${recorder_basename}
else
    echo "not LOCAL nor PUBLIC"
fi
