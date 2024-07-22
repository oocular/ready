#!/bin/bash

### USAGE
#conda activate readyVE (in local machine)
#bash frames_to_mp4_to_gxf.bash sample-frames png 30 validation-026580-640widthx400height 640 400 3

### INPUT ARGUMENTS
FRAMES_PATH=$1
IMAGE_EXTENSION=$2 # png or jpg
NNframes=$3
FRAMEFILENAMEID=$4
WIDTH=$5
HEIGHT=$6
CHANNELS=$7

### (1) CREATE DUPLICATE IMAGES TO CREATE video_???_duplicated_frames_of_???
mkdir -p ${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}
i=0; while ((i++ < ${NNframes})); do cp ../${FRAMES_PATH}/$FRAMEFILENAMEID.${IMAGE_EXTENSION} ${NNframes}_duplicated_frames_of_$FRAMEFILENAMEID/$FRAMEFILENAMEID"_$i."${IMAGE_EXTENSION}; done
ffmpeg -framerate ${NNframes} -pattern_type glob -i "${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}/*."${IMAGE_EXTENSION} -c:v libx264 video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}.mp4
ffmpeg -i video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}.mp4  #check video
rm -rf ${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID} #remove path of copied images
##vlc video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}.mp4  #check video


### (2) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
#### run #bash download-holohub-utils.bash at /src/ready/apis/holoscan/utils
ffmpeg -i video_${NNframes}_duplicated_frames_of_$FRAMEFILENAMEID.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename video_${NNframes}_duplicated_frames_of_$FRAMEFILENAMEID

echo "COMPLETION OF " ${NNframes}_duplicated_frames_of_$FRAMEFILENAMEID/$FRAMEFILENAMEID
