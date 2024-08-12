# Holoscan


## Build
```
#TODO: Update https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v2.3.0
cd $HOME/repositories
git clone https://github.com/nvidia-holoscan/holohub.git && cd holohub
./run clear_cache
#./dev_container build --verbose
./dev_container build --docker_file $HOME/Desktop/nystagmus-tracking/ready/docs/holoscan/Dockerfile #[+] Building 3470.5s (9/9) FINISHE
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

