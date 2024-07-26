#!/bin/bash

### USAGE
#cd ~/ready/data/bash-scripts
#conda activate readyVE #(in local machine)
#bash mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/novel/videos cut_video_640x400.mp4 640 400 3 24
#bash mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/novel/videos cut_video_640x400_grayscale.mp4 640 400 3 24

### INPUT ARGUMENTS
VIDEOPATH=$1
VIDEONAME=$2
WIDTH=$3
HEIGHT=$4
CHANNELS=$5
NNframes=$6

#FRAMES_PATH=$1
#IMAGE_EXTENSION=$2 # png or jpg
#FRAMEFILENAMEID=$4
echo ${VIDEOPATH}/${VIDEONAME}
##${FILE%%.*} filename
##${FILE#*.} extension

### Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
#### run #bash download-holohub-utils.bash at /src/ready/apis/holoscan/utils
ffmpeg -i ${VIDEOPATH}/${VIDEONAME} -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename ${VIDEOPATH}/${VIDEONAME%%.*}

echo Conversion completed ${VIDEONAME}!!
