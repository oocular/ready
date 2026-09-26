# Video Processing API
Convert and replay video recordings through the ready pipeline.


## Get started in 3 steps:

```bash
# 1. Launch the dev container
cd $HOME/repositories/oocular/ready/docs/holoscan
bash launch_dev_container.bash

# 2. Enter the workspace
cd /workspace/volumes/ready

# 3. Replay a recording
bash scripts/apis/webrtc_ready.bash config_webrtc_ready_template.yaml LOCAL DEBUG replayer_raw False
bash scripts/apis/webrtc_ready.bash config_webrtc_ready_template.yaml LOCAL DEBUG replayer_inference False
```

## convert_video_to_gxf_entities

### launch container
```bash 
cd $HOME/repositories/oocular/ready/docs/holoscan 
bash launch_dev_container.bash 
cd /workspace/volumes/ready
```

### Replay recordings

* Edit config
```bash
## Terminal Outside the container 
# Change to repo path
cd $HOME/repositories/oocular/ready/
# Setup config file
CONFIG_YAML=config_webrtc_ready_template.yaml
# Edit file if needed
vim configs/apis/${CONFIG_YAML}
```

* Replay recordings
```bash
# Setup config file
CONFIG_YAML=config_webrtc_ready_template.yaml
# Raw replay
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG replayer_raw False

# Inference replay
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG replayer_inference False
```

### convert gxf to mp4
```bash
cd /workspace/volumes/ready
bash scripts/video_processing/convert_gxf_entities_to_video.bash config_webrtc_ready_processing_template.yaml
# Guessed frame rate: 30.34931459907574 fps
# Frame array shape: 480x640x3 (height x width x channels)
```



## Process videos

The video processing step transforms raw camera input into a cropped and resized region of interest (ROI) suitable for downstream inference and streaming.

The diagram below illustrates the processing geometry:
* Blue rectangle, 640W × 480H: the native camera resolution.
* Red rectangle, 640W × 400H: the model input size.
* Green rectangle, 520W × 300H: the cropped and resized region of interest (ROI).
[fig](../figs/videoprocessing.svg)!

### Launch the development container
```bash 
cd $HOME/repositories/oocular/ready/docs/holoscan 
bash launch_dev_container.bash 
cd /workspace/volumes/ready
```

### Run the video processing script
```bash
cd /workspace/volumes/ready
bash scripts/video_processing/postprocessing.bash config_webrtc_ready_processing_template.yaml
```


##  View the result
Play the generated MP4 with ffplay:
```bash
ffplay -vf \
    "drawtext=text='%{n}':x=10:y=10:fontsize=40:fontcolor=blue, \
    drawtext=text='%{pts\\:hms}':x=10:y=50:fontsize=40:fontcolor=blue" \
    -autoexit *.mp4
```