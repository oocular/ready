## USAGE
# bash analyse_logfile.bash <PATH> <logger_video087.log>
PATHLOG=$1
LOGFILENAME=${2::-4} #removes extension
echo "LOGFILENAME: $LOGFILENAME"
echo "PATHLOG: $PATHLOG"

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate

python src/ready/apis/holoscan/utils/analyze.py -g $PATHLOG/$LOGFILENAME.log group_$LOGFILENAME --stddev -m -a --save-csv
mv avg_values.csv $PATHLOG/avg_values_$LOGFILENAME.csv && mv stddev_values.csv $PATHLOG/stddev_values_$LOGFILENAME.csv && mv max_values.csv $PATHLOG/max_values_$LOGFILENAME.csv
