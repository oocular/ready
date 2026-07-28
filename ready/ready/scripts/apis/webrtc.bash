#!/bin/bash
set -Ee
## USAGE
# bash webrtc.bash <$1:LOGGER_NAME.log> <$2:NET: LOCAL/PUBLIC> <$3:HOLOSCAN_LOG_LEVEL: OFF/DEBUG/TRACE/INFO/ERROR> <$4:SOURCE: webrtc_client/replayer> <$5:ENABLE_RECORDING: True/False>

##TODO
# https://rsalveti.wordpress.com/2007/04/03/bash-parsing-arguments-with-getopts/
# https://aplawrence.com/Unix/getopts.html
# https://unix.stackexchange.com/questions/677507/how-to-check-arguments-given-to-a-bash-script-efficiently


SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../

source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/apis/config_webrtc.yaml)
cd ${workspace_apiPath}

if [[ $2 == LOCAL ]]; then
    echo $2 LOCAL network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --logger_filename ${workspace_keysPath}/$1 --source $4 --enable_recording $5 --recording_directory ${recorder_directory} --recording_basename ${recorder_basename}
elif [[ $2 == PUBLIC ]]; then
    echo $2 PUBLIC network
    export HOLOSCAN_LOG_LEVEL=$3
    clear && python webrtc_client.py --cert-file ${workspace_keysPath}/MyCertificate.crt --key-file ${workspace_keysPath}/MyKey.key --logger_filename ${workspace_keysPath}/$1 --source $4 --enable_recording $5 --recording_directory ${recorder_directory} --recording_basename ${recorder_basename}
else
    echo "not LOCAL nor PUBLIC"
fi
