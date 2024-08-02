#!/bin/bash

### USAGE
#cd ~/ready/data/bash-scripts
#conda activate readyVE #(in local machine)

#bash frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/videos $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/sample-frames png 30 val-000180-640wX400h 640 400 3
#bash frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/videos $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/sample-frames png 30 four-frames-in-one 640 400 3
#bash frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/videos $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/sample-frames png 30 fractal-frames-in-one 640 400 3

### INPUT ARGUMENTS
VIDEOPATH=$1
FRAMES_PATH=$2
IMAGE_EXTENSION=$3 # png or jpg
NNframes=$4
FRAMEFILENAMEID=$5
WIDTH=$6
HEIGHT=$7
CHANNELS=$8

#### (1) CREATE DUPLICATE IMAGES TO CREATE video_???_duplicated_frames_of_???
mkdir -p ${VIDEOPATH}/${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}
i=0; while ((i++ < ${NNframes})); do cp ${FRAMES_PATH}/$FRAMEFILENAMEID.${IMAGE_EXTENSION} ${VIDEOPATH}/${NNframes}_duplicated_frames_of_$FRAMEFILENAMEID/$FRAMEFILENAMEID"_$i."${IMAGE_EXTENSION}; done
ffmpeg -y -framerate ${NNframes} -pattern_type glob -i "${VIDEOPATH}/${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}/*."${IMAGE_EXTENSION} -c:v libx264 ${VIDEOPATH}/video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}.mp4
ffmpeg -i ${VIDEOPATH}/video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}.mp4  #check video
rm -rf ${VIDEOPATH}/${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID} #remove path of copied images
###vlc video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}.mp4  #check video


### (2) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
#### run #bash download-holohub-utils.bash at /src/ready/apis/holoscan/utils
cd ../../src/ready/apis/holoscan/utils/
bash download-holohub-utils.bash
cd ../../../../../data/bash-scripts/
ffmpeg -y -i ${VIDEOPATH}/video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename ${VIDEOPATH}/video_${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}


echo "COMPLETION OF " ${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}
