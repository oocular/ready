#!/bin/bash
set -Eeuxo pipefail

### USAGE
#cd ~/ready
#source .venv/bin/activate
#cd ~/ready/data/bash-scripts

# bash mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos cut_video_640x400_7117d0.mp4 640 400 3 24
# bash mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos cut_video_640x400_cc6b03.mp4 640 400 3 24

### INPUT ARGUMENTS
VIDEOPATH=$1 # Path to the video
VIDEONAME=$2 # Name of the video including extension
WIDTH=$3 # Width of the video
HEIGHT=$4 # Height of the video
CHANNELS=$5 # Number of channels in the video
NNframes=$6 # Number of frames per second which should match the same FPS as the original video

echo ${VIDEOPATH}/${VIDEONAME}

## TODO: include the following arguments
#FRAMES_PATH=$1
#IMAGE_EXTENSION=$2 # png or jpg
#FRAMEFILENAMEID=$4
##${FILE%%.*} filename
##${FILE#*.} extension

### Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
#### run #bash download-holohub-utils.bash at /src/ready/apis/holoscan/utils
ffmpeg -i ${VIDEOPATH}/${VIDEONAME} -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename ${VIDEOPATH}/${VIDEONAME%%.*}

echo Conversion completed ${VIDEONAME}!!
