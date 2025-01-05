# Testing and processing videos

## Acute_VN_Left.MPG
```
ffmpeg -i Acute_VN_Left.MPG
  Stream #0:0[0x1e0]: Video: mpeg1video, yuv420p(tv), 320x240 [SAR 1:1 DAR 4:3], 1024 kb/s, 25 fps, 25 tbr, 90k tbn, 25 tbc
```
## Preprocess vidoeo
* Scale video to 640x400
```
VIDEO=Acute_VN_Left.MPG
cd $HOME/datasets/ready/videos/shared
ffmpeg -y -i ${VIDEO} -vf scale=640:400 video_640x400_${VIDEO}
# bash mp4_to_gxf
bash ../bash-scripts/mp4_to_gxf.bash $HOME/datasets/ready/videos/shared video_640x400_${VIDEO} 640 400 3 30
```
* Crop and Scale video to 640x400
```
# To crop a W250×H200 section, starting from position (X00, Y100)
# ffplay -i ${VIDEO} -vf "crop=250:200:00:100" #to_test
ffmpeg -i ${VIDEO} -vf "crop=100:100:50:100" -c:a copy cropped_${VIDEO}
ffmpeg -y -i cropped_${VIDEO} -vf scale=640:400 cropped_video_640x400_${VIDEO}
# bash mp4_to_gxf
bash ../bash-scripts/mp4_to_gxf.bash $HOME/datasets/ready/videos/shared cropped_video_640x400_${VIDEO} 640 400 3 30
```

