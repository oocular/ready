## USAGE
# bash analyse_logfile.bash logger_video087
sudo apt-get install xdot

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

xdot live_app_graph.dot


