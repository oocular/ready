#!/bin/bash
set -Eeuxo pipefail

cd $HOME/datasets/ready/videos/benign-positional-vertigo

VIDEO1=video1-right-posterior-canalithiasis.mp4
if [ -f "${VIDEO1}" ];
then
  echo "${VIDEO1} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc1.mp4 -O ${VIDEO1}
fi

VIDEO2=video2-left-posterior-cupulolithiasis.mp4
if [ -f "${VIDEO2}" ];
then
  echo "${VIDEO2} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc2.mp4 -O ${VIDEO2}
fi

VIDEO3=video3-right-anterior-canalithiasis.mp4
if [ -f "${VIDEO3}" ];
then
  echo "${VIDEO3} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc3.mp4 -O ${VIDEO3}
fi

VIDEO4=video4-right-lateral-canalithiasis.mp4
if [ -f "${VIDEO4}" ];
then
  echo "${VIDEO4} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc4.mp4 -O ${VIDEO4}
fi

VIDEO5=video5-right-lateral-cupololithiasis.mp4
if [ -f "${VIDEO5}" ];
then
  echo "${VIDEO5} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc5.mp4 -O ${VIDEO5}
fi

VIDEO6=video6-menieres-disease.mp4
if [ -f "${VIDEO6}" ];
then
  echo "${VIDEO6} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc6.mp4 -O ${VIDEO6}
fi

VIDEO7=video7-vestibular-migraine.mp4
if [ -f "${VIDEO7}" ];
then
  echo "${VIDEO7} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc7.mp4 -O ${VIDEO7}
fi

VIDEO8=video8-cerebellar-arteriovenous-malformation.mp4
if [ -f "${VIDEO8}" ];
then
  echo "${VIDEO8} file exist!"
else 
  wget https://ars.els-cdn.com/content/image/1-s2.0-S2467981X19300125-mmc8.mp4 -O ${VIDEO8}
fi


### Crop and scale videos of 1280x720
## To crop a W415×H250 section, starting from position (X00, Y400)
# ffplay -i video1-right-posterior-canalithiasis.mp4 -vf "crop=415:250:435:30" #to_test
ffmpeg -i ${VIDEO1} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO1}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO1} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO1}
rm cropped_WxH_cut_video_640x400_${VIDEO1}

ffmpeg -i ${VIDEO2} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO2}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO2} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO2}
rm cropped_WxH_cut_video_640x400_${VIDEO2}

ffmpeg -i ${VIDEO3} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO3}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO3} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO3}
rm cropped_WxH_cut_video_640x400_${VIDEO3}

ffmpeg -i ${VIDEO4} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO4}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO4} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO4}
rm cropped_WxH_cut_video_640x400_${VIDEO4}

ffmpeg -i ${VIDEO6} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO6}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO6} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO6}
rm cropped_WxH_cut_video_640x400_${VIDEO6}

ffmpeg -i ${VIDEO7} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO7}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO7} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO7}
rm cropped_WxH_cut_video_640x400_${VIDEO7}

ffmpeg -i ${VIDEO8} -vf "crop=415:250:435:30" -c:a copy cropped_WxH_cut_video_640x400_${VIDEO8}
ffmpeg -y -i cropped_WxH_cut_video_640x400_${VIDEO8} -vf scale=640:400 cropped_cut_video_640x400_${VIDEO8}
rm cropped_WxH_cut_video_640x400_${VIDEO8}



#VIDEOPATH=$1
#VIDEOLINK=$2
#START_TIME=$3
#DURATION_FRAME=$4
#VIDEO_SHORTNAME=${VIDEOLINK:0-10}

### Scale video to 640x400
#ffmpeg -y -i ${VIDEOPATH}/${VIDEO_SHORTNAME} -vf scale=640:400 ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME}
#
### Cut timeframes of the video
##>The process begins at 0 minute, 51 seconds as indicated by the “-ss” flag.
##>This will cut a 5 -seconds indicated at the “-t” flag, which is the duration. 
##>-acodec copy and -vcodec copy copy the codec data without transcoding (which would incur quality loss).
##>https://stackoverflow.com/questions/15629490/how-to-cut-part-of-mp4-video-using-ffmpeg-without-losing-quality
#
#ffmpeg -y -ss ${START_TIME}  -i  ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME} -vcodec libx264 -acodec copy -t ${DURATION_FRAME} ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME}
#

### Convert video to grayscale and hue
##ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME} -vf format=gray ${VIDEOPATH}/cut_video_640x400_grayscale_${VIDEO_SHORTNAME}
##ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400_${VIDEO_SHORTNAME} -vf hue=s=20 ${VIDEOPATH}/cut_video_640x400_hue_${VIDEO_SHORTNAME}
#
### Remove files
#rm ${VIDEOPATH}/cropped_WxH_cut_video_640x400_${VIDEO_SHORTNAME}
#rm ${VIDEOPATH}/${VIDEO_SHORTNAME}
#rm ${VIDEOPATH}/video_640x400_${VIDEO_SHORTNAME}
