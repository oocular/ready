#USAGE
#bash conver-grayscale-to-rgb.bash val-000160-640wX400h

IN=$1
#OUT=$2

convert ${IN}.png -define png:color-type=2 ${IN}_rgb.png

