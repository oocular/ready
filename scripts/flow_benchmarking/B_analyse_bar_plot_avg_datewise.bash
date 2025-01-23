## USAGE
# bash analyse_logfile.bash logger_video087
LOGF01=$1
LOGF02=$2
#LOGF03=$3
#LOGF04=$4
#LOGF05=$5
PATHLOG=$3
echo "LOGFILENAME: $LOGFILENAME"
echo "PATHLOG: $PATHLOG"

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

python src/ready/apis/holoscan/utils/analyze.py --stddev -m -a --save-csv --draw-cdf --no-display-graphs \
-g $PATHLOG/$LOGF01.log $LOGF01 \
-g $PATHLOG/$LOGF02.log $LOGF02


mv avg_values.csv $PATHLOG && mv stddev_values.csv $PATHLOG && mv max_values.csv $PATHLOG
mv cdf_curve.png $PATHLOG/cdf_curve_$LOGF01.png 
#rm avg_values.csv max_values.csv stddev_values.csv  


###for two logs
python src/ready/apis/holoscan/utils/bar_plot_avg_datewise.py $PATHLOG/avg_values.csv $PATHLOG/stddev_values.csv
mv avg_* $PATHLOG/avg_plot_$LOGF01.png
#python src/ready/apis/holoscan/utils/bar_plot_avg_datewise.py \
#$PATHLOG/avg_values_$LOGF01.csv \
#$PATHLOG/avg_values_$LOGF02.csv \
#$PATHLOG/stddev_values_$LOGF01.csv \
#$PATHLOG/stddev_values_$LOGF02.csv


