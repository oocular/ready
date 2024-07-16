# Holoscan

## Build
```
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub
./run clear_cache
./dev_container build --verbose
```

## Launch 
```
cd $HOME/repositories/holohub
./dev_container launch --add-volume $HOME/Desktop/nystagmus-tracking/ready
```

## Debugging
### v4l2_camera
```
cd /workspace/volumes/ready/src/ready/apis/holoscan/v4l2_camera/python
python v4l2_camera.py
```

### bring your own model
```
cd /workspace/volumes/ready/src/ready/apis/holoscan/v4l2_camera/python
python byom.py --data /workspace/volumes/ready/data/openEDS/videos
```

