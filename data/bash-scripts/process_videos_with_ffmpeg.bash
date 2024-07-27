#cd ~/ready/data/bash-scripts
#conda activate readyVE #(in local machine)
# bash process_videos_with_ffmpeg.bash $HOME/Desktop/nystagmus-tracking/ready/data/novel/videos


VIDEOPATH=$1
#VIDEONAME=$2


#The process begins at 0 minute, 51 seconds as indicated by the “-ss” flag.
#This will cut a 5 -seconds indicated at the “-t” flag, which is the duration. 
#-acodec copy and -vcodec copy copy the codec data without transcoding (which would incur quality loss).
#https://stackoverflow.com/questions/15629490/how-to-cut-part-of-mp4-video-using-ffmpeg-without-losing-quality


## VIDEO ref
# https://collections.lib.utah.edu/details?id=1213434&q=creator_t%3A%22gold%22+AND+curriculum_t%3A%22gazeevokednys%22&fd=title_t%2Cdescription_t%2Csubject_t&sort=facet_title_t+asc&facet_setname_s=ehsl_novel_gold

if [ -f "${VIDEOPATH}/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4" ];
then
  echo "/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4 file exist!"
else 
  wget https://collections.lib.utah.edu/dl_files/f3/52/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4 -O ${VIDEOPATH}/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4
fi

ffmpeg -y -i ${VIDEOPATH}/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4 -vf scale=640:400 ${VIDEOPATH}/video_640x400.mp4
#ffmpeg -ss 0:52  -i  video_640x400.mp4 -codec copy -t 0:10 cut_video_640x400.mp4
#ffmpeg -ss 0:52  -i  video_640x400.mp4 -acodec copy -t 0:10 cut_video_640x400.mp4

ffmpeg -y -ss 0:52  -i  ${VIDEOPATH}/video_640x400.mp4 -vcodec libx264 -acodec copy -t 0:10 ${VIDEOPATH}/cut_video_640x400.mp4
ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400.mp4 -vf format=gray ${VIDEOPATH}/cut_video_640x400_grayscale.mp4
#ffmpeg -y -i ${VIDEOPATH}/cut_video_640x400.mp4 -vf hue=s=20 ${VIDEOPATH}/cut_video_640x400_hue.mp4

rm ${VIDEOPATH}/video_640x400.mp4
rm ${VIDEOPATH}/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4
