#!/bin/bash
set -Eeuxo pipefail

### USAGE
#cd ~/ready/data/bash-scripts
#conda activate readyVE #(in local machine)
#bash multi_frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/sample-frames/val3frames/images $HOME/Desktop/nystagmus-tracking/ready/data/openEDS/videos 000160 000170 000180 png
#bash multi_frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/sample-frames/test640x400/images $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos 1_1i_Ls_1 1_1i_Lr_2 1_1i_Ll_2 jpg video_3framesx10


### (0) SETTING UP VARIABLES
DATA_PATH=$1
VIDEO_PATH=$2
IMAGE_ID_1=$3 #1_1i_Ls_1 #(center)
IMAGE_ID_2=$4 #1_1i_Lr_2 #(left)
IMAGE_ID_3=$5 #1_1i_Ll_2 #(rigth)
IMAGE_EXTENSION=$6 #png or jpg
VIDEO_NAME=$7 #video_3framesx10
ID_PATH=temp
mkdir -p ${VIDEO_PATH}/${ID_PATH}


##### (1.A) CREATE DUPLICATE IMAGES TO CREATE video_???_duplicated_frames_of_???
ReplicatedFrames=10
i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/${IMAGE_ID_1}.${IMAGE_EXTENSION} ${VIDEO_PATH}/${ID_PATH}/${IMAGE_ID_1}"_$i."${IMAGE_EXTENSION}; done

i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/${IMAGE_ID_2}.${IMAGE_EXTENSION} ${VIDEO_PATH}/${ID_PATH}/${IMAGE_ID_2}"_$i."${IMAGE_EXTENSION}; done

i=0; while ((i++ < ${ReplicatedFrames})); do cp ${DATA_PATH}/${IMAGE_ID_3}.${IMAGE_EXTENSION} ${VIDEO_PATH}/${ID_PATH}/${IMAGE_ID_3}"_$i."${IMAGE_EXTENSION}; done



######TODO (1.B) CREATE RANDOM ${NUMBER_OF_IMAGES} from data to create a path of images
#NUMBER_OF_IMAGES=30
#shuf -n ${NUMBER_OF_IMAGES} -e ${DATA_PATH}/*.png | xargs -i mv {} ${VIDEO_PATH}/${ID_PATH}



#### (2) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device


### (2.A) INPUT ARGUMENTS FOR MP4 to GXF
#VIDEOPATH=$1
#FRAMES_PATH=$2
NNframes=30
#FRAMEFILENAMEID=$5
WIDTH=640
HEIGHT=400
CHANNELS=3

### CONVERT TO VIDEO
ffmpeg -y -framerate ${NNframes} -pattern_type glob -i "${VIDEO_PATH}/${ID_PATH}/*.${IMAGE_EXTENSION}" -c:v libx264 ${VIDEO_PATH}/${VIDEO_NAME}.mp4
echo "SUCESSFUL CONVERSION OF " ${VIDEO_PATH}/${VIDEO_NAME}.mp4

##ffmpeg -i ${VIDEO_PATH}/${VIDEO_NAME}.mp4


###### (2.B) Converting mp4 to gxf (Graph Execution Framework) using `ffmeg` and `convert_video_to_gxf_entities.py` in local host device
cd ../../src/ready/apis/holoscan/utils/
bash download-holohub-utils.bash
cd ../../../../../data/bash-scripts/
echo $pwd
ffmpeg -y -i ${VIDEO_PATH}/${VIDEO_NAME}.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python ../../src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py --width ${WIDTH} --height ${HEIGHT} --channels ${CHANNELS} --framerate ${NNframes} --basename ${VIDEO_PATH}/${VIDEO_NAME}
echo "COMPLETION OF " ${NNframes}_duplicated_frames_of_${FRAMEFILENAMEID}_channels${CHANNELS}
rm -rf ${VIDEO_PATH}/${ID_PATH}

