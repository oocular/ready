# Holoscan


## Build
```
#TODO: Update https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v2.3.0
cd $HOME/repositories
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub
git pull
git checkout 1a67c53 #holoscan-sdk-2.0.0
./run clear_cache
#./dev_container build --verbose
#TOTEST./dev_container vscode --docker_file $HOME/Desktop/nystagmus-tracking/ready/docs/holoscan/Dockerfile #[+] Building 3470.5s #~1h (9/9) FINISHED
./dev_container build --docker_file $HOME/Desktop/nystagmus-tracking/ready/docs/holoscan/Dockerfile #[+] Building 3470.5s #~1h (9/9) FINISHED
```

## Launch 
```
cd $HOME/Desktop/nystagmus-tracking/ready/docs/holoscan
bash launch_dev_container.bash
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

### Issue
* issue
```
docker run --net host --interactive --tty -u 1000:1000 -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /home/mxochicale/repositories/holohub:/workspace/holohub -w /workspace/holohub --runtime=nvidia --gpus all --cap-add CAP_SYS_PTRACE --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/mxochicale/Desktop/nystagmus-tracking/ready:/workspace/volumes/ready -v /home/mxochicale/repositories/holoscan-sdk:/workspace/volumes/holoscan-sdk --device /dev/video0 --device /dev/video1 --device /dev/video2 --device /dev/video3 --device /dev/snd/timer --device /dev/snd/seq --device /dev/snd/pcmC0D0p --device /dev/snd/pcmC0D0c --device /dev/snd/pcmC0D3p --device /dev/snd/pcmC0D7p --device /dev/snd/pcmC0D8p --device /dev/snd/pcmC0D9p --device /dev/snd/hwC0D0 --device /dev/snd/hwC0D2 --device /dev/snd/controlC0 -v /etc/asound.conf:/etc/asound.conf --group-add 29 -e DISPLAY --group-add video --rm -e CUPY_CACHE_DIR=/workspace/holohub/.cupy/kernel_cache --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 holohub:ngc-v2.0.0-dgpu
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: nvml error: driver not loaded: unknown.
```

* Remove all non-related images and stop IDs (don't work)
```
docker images
REPOSITORY   TAG               IMAGE ID       CREATED       SIZE
holohub      ngc-v2.2.0-dgpu   44a333fcfd9f   3 weeks ago   15.6GB

docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```
* Update and upgrate and reboot (dont' work)
```
sudo apt update
sudo apt upgrade
```

* remove and build docker image
```
git checkout 1a67c53 #holoscan-sdk-2.0.0
[+] Building 3467.2s (9/9) FINISHED  
                                                                                                                                               docker:default
 => [internal] load build definition from Dockerfile                                                                                                                                                          0.0s
 => => transferring dockerfile: 1.97kB                                                                                                                                                                        0.0s
 => WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 2)                                                                                                                                0.0s
 => WARN: InvalidDefaultArgInFrom: Default value for ARG ${BASE_IMAGE} results in empty or invalid base image name (line 2)  


                                                                                                                      0.0s

 2 warnings found (use --debug to expand):
 - FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 2)
 - InvalidDefaultArgInFrom: Default value for ARG ${BASE_IMAGE} results in empty or invalid base image name (line 2)


```

* open an issue

https://github.com/nvidia-holoscan/holohub/issues/479 




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
