## USAGE
# bash analyse_logfile.bash logger_video087
PATHLOG=$1
LOGF01=${2::-4} #removes extension
LOGF02=${3::-4} #removes extension
LOGF03=${4::-4} #removes extension
LOGF04=${5::-4} #removes extension
LOGF05=${6::-4} #removes extension
echo "PATHLOG: $PATHLOG"
echo "LOGF01 $LOGF01"
echo "LOGF02 $LOGF02"
echo "LOGF03 $LOGF03"
echo "LOGF04 $LOGF04"
echo "LOGF05 $LOGF05"

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

python src/ready/apis/holoscan/utils/analyze.py --stddev -m -a --save-csv --draw-cdf --no-display-graphs \
-g $PATHLOG/$LOGF01.log $LOGF01 \
-g $PATHLOG/$LOGF02.log $LOGF02 \
-g $PATHLOG/$LOGF03.log $LOGF03 \
-g $PATHLOG/$LOGF04.log $LOGF04 \
-g $PATHLOG/$LOGF05.log $LOGF05

mv avg_values.csv $PATHLOG && mv stddev_values.csv $PATHLOG && mv max_values.csv $PATHLOG
mv cdf_curve.png $PATHLOG/cdf_curve_$LOGF01.png
rm $PATHLOG/avg_values.csv $PATHLOG/max_values.csv $PATHLOG/stddev_values.csv

python src/ready/apis/holoscan/utils/bar_plot_avg_datewise.py \
$PATHLOG/avg_values_$LOGF01.csv \
$PATHLOG/avg_values_$LOGF02.csv \
$PATHLOG/avg_values_$LOGF03.csv \
$PATHLOG/avg_values_$LOGF04.csv \
$PATHLOG/avg_values_$LOGF05.csv \
$PATHLOG/stddev_values_$LOGF01.csv \
$PATHLOG/stddev_values_$LOGF02.csv \
$PATHLOG/stddev_values_$LOGF03.csv \
$PATHLOG/stddev_values_$LOGF04.csv \
$PATHLOG/stddev_values_$LOGF05.csv

mv avg_* $PATHLOG/avg_plot_$LOGF01.png
