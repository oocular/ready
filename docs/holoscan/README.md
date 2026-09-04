# NVIDIA Holoscan SDK [:link:](https://developer.nvidia.com/holoscan-sdk)

## Requirements

* install CUDA drivers 
```bash
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

* install v4l
```bash
sudo apt-get install v4l-utils
```

## Clone holoscan-sdk (useful for `VideoStreamRecorderOp, VideoStreamReplayerOp`)
```bash
cd $HOME/repositories
git clone git@github.com:nvidia-holoscan/holoscan-sdk.git
cd $HOME/repositories/holoscan-sdk
git checkout main
git pull origin main
git tag #check tags
git checkout v4.6.0
./run clear_cache
./run build_image # produces holoscan-sdk-build-x86_64:<tag>
docker system prune -f --volumes #clean unused systems
docker images 
# holoscan-sdk-build-cu12-x86_64:latest      dcfe0ca2d68d       20.3GB            7GB        
##v2.2.0
# git checkout v2.2.0 # Use `git tag` to discover other available versions
```

## Build image with specific version
```bash
## [First time] Clone repo
cd $HOME/repositories
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub

## Already cloned
cd $HOME/repositories/holohub
git checkout main
git pull
git tag #check tags
# git checkout holoscan-sdk-3.2.0
# ./run clear_cache
# ./dev_container build --docker_file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile #[+] Building 452.4s (8/8) FINISHED
git checkout holoscan-sdk-4.6.0
./holohub build-container --help
#TODO: check if `ready/docs/holoscan/Dockerfile` is needed as it might cover it in the lastest versions
./holohub build-container --docker-file $HOME/repositories/oocular/ready/docs/holoscan/Dockerfile --base-img holoscan-sdk-build-cu12-x86_64:latest
docker system prune -f --volumes #clean unused systems
docker images
./holohub run-container
docker system prune -f --volumes #clean unused systems
#
#
##logs
#git checkout 1a67c53 #holoscan-sdk-2.0.0
#git checkout 3834a7b #holoscan-sdk-2.5.0 #WORKS! pointing to "holohub:ngc-v2.4.0" > https://github.com/nvidia-holoscan/holohub/blob/3834a7b057501d6dbc564df05692866d2b775324/dev_container#L472
#git checkout f7f561f #holoscan-sdk-2.6.0 #WORKS! pointing to "holohub:ngc-v2.5.0" [+] Building holoscan-sdk-2.6.0 2997.6s (8/8) FINISHED #~50mins
#git checkout 9554bd3 #holoscan-sdk-2.7.0 #ISSUES! Wed Dec 4 10:26:06 2024 -0500
#git checkout 9ce2638 #holoscan-sdk-2.8.0 Thu Jan 2 16:32:07 2025 -0500
#git checkout holoscan-sdk-2.9.0 Mon Jan 27 12:57:12 2025 -0800
#git checkout holoscan-sdk-3.0.0.7 #Sat  1 Mar 18:34:41 GMT 2025
#git checkout holoscan-sdk-3.0.0 #Thu 20 Mar 21:57:14 GMT 2025
#git checkout holoscan-sdk-3.4.0 #Sun 27 Jul 15:09:08 BST 2025 #=> ERROR [3/4] RUN chmod +rwx /usr/bin/python3.10
#
#
#TOTEST
#./dev_container vscode --docker_file $PATH/Dockerfile
```

## Docker image version and size
```bash
$ docker images
IMAGE                                    ID             DISK USAGE   CONTENT SIZE   EXTRA
holohub:ngc-v3.2.0-dgpu                  3ec1e840fdbe       26.6GB          8.8GB  
```

## Register the runtime with Docker
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker # Restart Docker
```

## Quick test

* Launch dev container
```bash
cd $HOME/repositories/oocular/ready/docs/holoscan
bash launch_dev_container.bash
```

* Run v4l2_camera
```bash
cd /workspace/volumes/ready/scripts/apis #cd /workspace/volumes/ready/src/ready/apis/holoscan/v4l2_camera/python
bash v4l2_cam.bash
```

* Edit v4l device number
```bash
vim src/ready/apis/holoscan/v4l2_camera/python/v4l2_camera.yaml
#  default device is /dev/video0" but for logitech in rtx2000 8gb gpu is device "/dev/video48"
* Exit
To exit dev container image, just type exit in the launched container.


## Run APIS

* [apis](apis.md): v4l2_camera, Bring Your Own Model
* [apis_ready](apis_ready.md): READY 
* [apis_webrtc](apis_webrtc.md)
* [apis_webrtc_ready](apis_webrtc_ready.md)

## Docker commands
```bash
docker images
docker ps
docker attach <ID> e.g. `docker attach $(docker ps -aq)`
docker stop <ID> e.g. `docker stop $(docker ps -aq)`
docker rename keen_einstein mycontainer
docker rmi --force <ID>

docker stop $(docker ps -a -q)
docker system prune -f --volumes #clean unused systems
```


