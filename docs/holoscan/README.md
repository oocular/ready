# Holoscan

## Requiremetns
```
#install cuda 
https://github.com/mxochicale/code/blob/main/gpu/installation/installing_cuda.bash
#checking driver version
nvidia-smi

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit #solves 2025-01-04 22:20:45 [FATAL] nvidia-ctk not found. Please install the NVIDIA Container Toolkit.
#docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
sudo apt install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart dock
```

## Build
```
#first time clone
cd $HOME/repositories
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub
# alrady cloned
cd $HOME/repositories/holohub
git pull
git tag
git checkout holoscan-sdk-2.9.0
./run clear_cache
./dev_container build --docker_file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile #[+] Building 347.9s (9/9) FINISHED
#logs
#git checkout 1a67c53 #holoscan-sdk-2.0.0
#git checkout 3834a7b #holoscan-sdk-2.5.0 #WORKS! pointing to "holohub:ngc-v2.4.0" > https://github.com/nvidia-holoscan/holohub/blob/3834a7b057501d6dbc564df05692866d2b775324/dev_container#L472
#git checkout f7f561f #holoscan-sdk-2.6.0 #WORKS! pointing to "holohub:ngc-v2.5.0" [+] Building holoscan-sdk-2.6.0 2997.6s (8/8) FINISHED #~50mins
#git checkout 9554bd3 #holoscan-sdk-2.7.0 #ISSUES! Wed Dec 4 10:26:06 2024 -0500
#git checkout 9ce2638 #holoscan-sdk-2.8.0 Thu Jan 2 16:32:07 2025 -0500
#git checkout holoscan-sdk-2.9.0 Mon Jan 27 12:57:12 2025 -0800
#TOTEST
#./dev_container vscode --docker_file $PATH/Dockerfile
```

## Run and debug

See [apis](apis.md)


## Docker commands
```
docker images
docker ps
docker attach <ID>
docker stop <ID>
docker rename keen_einstein mycontainer
docker rmi --force <ID>
```


## v4l2


* /dev/video0
```

v4l2-ctl -d /dev/video0 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1280x720
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x180
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
```

* /dev/video4

```
 v4l2-ctl -d /dev/video4 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

...

		Size: Discrete 2304x1536
			Interval: Discrete 0.500s (2.000 fps)
	[1]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.042s (24.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
			Interval: Discrete 0.100s (10.000 fps)
			Interval: Discrete 0.133s (7.500 fps)
			Interval: Discrete 0.200s (5.000 fps)

```


* USB endoscope camera (1/9 inch sensor size; 30fps; 70CAngleView)
```
v4l2-ctl -d /dev/video4 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 352x288
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 176x144
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)

```
