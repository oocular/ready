# Video convertion from `852x480` to `640 x 400`

## f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4 to cut_video_640x400.mp4
```
cd ~/ready/data/bash-scripts
#f352ab0b629842e9057f4122f5f75ac06a7117d0 > https://collections.lib.utah.edu/ark:/87278/s6s50fth 
#a2be1430eb699e9e53e418eb749883abe5cc6b03.mp4 > https://collections.lib.utah.edu/ark:/87278/s617xsxb
bash process_videos_with_ffmpeg.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos https://collections.lib.utah.edu/dl_files/f3/52/f352ab0b629842e9057f4122f5f75ac06a7117d0.mp4 0:52 0:10
bash process_videos_with_ffmpeg.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos https://collections.lib.utah.edu/dl_files/a2/be/a2be1430eb699e9e53e418eb749883abe5cc6b03.mp4 0:05 0:10
```

Reference 
https://collections.lib.utah.edu/details?id=1213434&q=creator_t%3A%22gold%22+AND+curriculum_t%3A%22gazeevokednys%22&fd=title_t%2Cdescription_t%2Csubject_t&sort=facet_title_t+asc&facet_setname_s=ehsl_novel_gold





