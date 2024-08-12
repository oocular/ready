#!/bin/bash

### USAGE
#cd ~/ready/data/bash-scripts
#conda activate readyVE #(in local machine)
#bash multi_frames_to_mp4_to_gxf.bash


### (0) SETTING UP VARIABLES
DATA_PATH=$HOME/Desktop/nystagmus-tracking/datasets/openEDS/openEDS/openEDS/validation/images
VIDEO_PATH=$HOME/Desktop/nystagmus-tracking/ready/data/openEDS/videos 
ID_PATH=temp
#cd $IMAGE_PATH
mkdir -p ${VIDEO_PATH}/${ID_PATH}
VIDEO_NAME=video_multiple_frames


##### (1.A) CREATE DUPLICATE IMAGES TO CREATE video_???_duplicated_frames_of_???
ReplicatedFrames=5
IMAGE_ID=014340
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done

IMAGE_ID=006110
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done

IMAGE_ID=007000
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done

IMAGE_ID=012700
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done

IMAGE_ID=023860
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done

IMAGE_ID=001210
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/$IMAGE_ID.png ${VIDEO_PATH}/${ID_PATH}/$IMAGE_ID"_$i."png; done


###### (1.B) CREATE RANDOM ${NUMBER_OF_IMAGES} from data to create a path of images
#NUMBER_OF_IMAGES=30
#shuf -n ${NUMBER_OF_IMAGES} -e ${DATA_PATH}/*.png | xargs -i mv {} ${VIDEO_PATH}/${ID_PATH}



#### (2) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device


#### (2.A) INPUT ARGUMENTS FOR MP4 to GXF
#VIDEOPATH=$1
#FRAMES_PATH=$2
#IMAGE_EXTENSION=$3 # png or jpg
NNframes=30
#FRAMEFILENAMEID=$5
WIDTH=640
HEIGHT=400
CHANNELS=3

## CONVERT TO VIDEO
ffmpeg -y -framerate ${NNframes} -pattern_type glob -i "${VIDEO_PATH}/${ID_PATH}/*.png" -c:v libx264 ${VIDEO_PATH}/${VIDEO_NAME}.mp4
#ffmpeg -i ${VIDEO_PATH}/${VIDEO_NAME}.mp4

#### (2.B) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
cd ../../src/ready/apis/holoscan/utils/
bash download-holohub-utils.bash
cd ../../../../../data/bash-scripts/
ffmpeg -y -i ${VIDEO_PATH}/${VIDEO_NAME}.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename ${VIDEO_PATH}/${VIDEO_NAME}

echo "COMPLETION OF " ${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}

rm -rf ${VIDEO_PATH}/${ID_PATH}

