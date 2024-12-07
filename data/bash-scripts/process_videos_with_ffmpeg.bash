#!/bin/bash
set -Eeuxo pipefail

VIDEOPATH=$1
VIDEOLINK=$2
START_TIME=$3
DURATION_FRAME=$4
VIDEO_SHORTNAME=${VIDEOLINK:0-10}


if [ -f "${VIDEOPATH}/${VIDEO_SHORTNAME}" ];
then
  echo "/${VIDEO_SHORTNAME} file exist!"
else 
  wget ${VIDEOLINK} -O ${VIDEOPATH}/${VIDEO_SHORTNAME}
fi


## Scale video to 640x400
ffmpeg -y -i ${VIDEOPATH}/${VIDEO_SHORTNAME} -vf scale=640:400 ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME}

## Cut timeframes of the video
#>The process begins at 0 minute, 51 seconds as indicated by the “-ss” flag.
#>This will cut a 5 -seconds indicated at the “-t” flag, which is the duration. 
#>-acodec copy and -vcodec copy copy the codec data without transcoding (which would incur quality loss).
#>https://stackoverflow.com/questions/15629490/how-to-cut-part-of-mp4-video-using-ffmpeg-without-losing-quality

ffmpeg -y -ss ${START_TIME}  -i  ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME} -vcodec libx264 -acodec copy -t ${DURATION_FRAME} ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME}

# ## Convert video to grayscale and hue
ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME} -vf format=gray ${VIDEOPATH}/cut_video_640x400_grayscale_${VIDEO_SHORTNAME}
ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME} -vf hue=s=20 ${VIDEOPATH}/cut_video_640x400_hue_${VIDEO_SHORTNAME}

# # ## Remove files
rm ${VIDEOPATH}/${VIDEO_SHORTNAME}
rm ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME}
