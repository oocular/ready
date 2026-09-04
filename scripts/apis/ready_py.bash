#!/bin/bash
set -Ee

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/apis/config_readypy_template.yaml)

cd ${workspace_apiPath}
#--source $1 #replayer #v4l2

clear && python ready.py --config_file ${workspace_configYamlPath}/${workspace_configYamlFile} --logger_filename ${recorder_directory}/${recorder_basename}_${recorder_loggername}_${recorder_loggerextension} --recording_directory ${recorder_directory} --recording_basename ${recorder_basename} --source $1
