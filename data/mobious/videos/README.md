# Videos

## Converting image to mp4 video to `gxf_entities`

### 1 Select image

![fig](../sample-frames/1_1i_Ll_1.jpg)

### 2 convert image to frames

```
cd ~/ready/data/bash-scripts
conda activate readyVE #(in local machine)
bash frames_to_mp4_to_gxf.bash $HOME/Desktop/nystagmus-tracking/ready/data/mobious/videos $HOME/Desktop/nystagmus-tracking/ready/data/mobious/sample-frames jpg 30 1_1i_Ll_1 640 400 3
```

### 3 video properties

```
ffmpeg -i video_30_duplicated_frames_of_1_1i_Ll_1_channels3.mp4 
#  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuvj420p(pc, bt470bg/unknown/unknown), 640x400 [SAR 1:1 DAR 8:5], 116 kb/s, 30 fps, 30 tbr, 15360 tbn, 60 tbc (default)
```




