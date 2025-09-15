# [NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk)

## Requirements
```bash
#install CUDA drivers 
cd ~/Downloads/
wget https://raw.githubusercontent.com/mxochicale/code/refs/heads/main/gpu/installation/installing_cuda.bash
bash installing_cuda.bash
#checking driver version
nvidia-smi

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit #solves 2025-01-04 22:20:45 [FATAL] nvidia-ctk not found. Please install the NVIDIA Container Toolkit.
#docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
sudo apt install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart dock

#REBOOT MACHINE!
```

## Build and update to latest version
```bash
## [First time] Clone repo
cd $HOME/repositories
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub

## Already cloned
cd $HOME/repositories/holohub
git checkout main
git pull
git tag #check tags
git checkout holoscan-sdk-3.6.0
./holohub clear-cache
./holohub build-container --no-cache --docker-file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile
## ALTERNATIVELY using VSCODE
# ./holohub vscode -h
# ./holohub vscode --docker-file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile #[+] Building 452.4s (8/8) FINISHED
```

## Run and debug

See [apis](apis.md)


## Docker commands
Clean up the container images:
```
docker system prune -f --volumes
docker images --format '{{.Repository}}:{{.Tag}}' | grep '^vsc-holohub' | xargs -r docker rmi
```

```
docker images
docker ps
docker attach <ID>
docker stop <ID>
docker rename keen_einstein mycontainer
docker rmi --force <ID>

docker stop $(docker ps -a -q)
docker system prune -f --volumes #clean unused systems
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
