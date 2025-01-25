## USAGE
# bash analyse_logfile.bash logger_video087

PATHLOGS=$1

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate

python src/ready/apis/holoscan/utils/app_perf_graph.py -o $PATHLOGS/live_app_graph.dot -l $PATHLOGS
