# Cricket machine: ARM CPU with A100 Nvidia GPU

## Features
* HD: 5TB
* GPU: NVIDIA A100, A100 , 40 GB HBM2, 1.6 TB/sec https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
* https://www.ucl.ac.uk/advanced-research-computing/coming-soon-platforms

## Connect
1. Connect to `vpn.ucl.ac.uk` using cisco https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
2. Connect to the server
```
ssh -X ccxxxxx@cricket.rc.ucl.ac.uk
xterm -rv & # open as many terminals you want
```

## Copying dataset
```
# openEDS.zip
scp openEDS.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS #openEDS.zip #8.0GB ETA 1h at 2MB/s
# MOBIUS.zip
scp MOBIUS.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious #MOBIUS.zip #3.3GB ETA  26mins at 2MB/s
scp strain-morbious.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious #34MB   1.9MB/s   00:17
```

## Container

### Setting up /opt/nvidia/containers (first time)
```
mkdir -p containers && cd containers
#cd containers
cp /opt/nvidia/containers/pytorch:24.04-y3.sif .
apptainer run pytorch:24.04-y3.sif
python
import torch
torch.cuda.is_available()

watch -n 2 nvidia-smi #in another terminal to see activity every 2secs
```

### Lauch container
```
# Updates repo
cd $HOME/ready
git pull

# Launch container 
bash docs/cricket/launch_container_in_cricket.bash

#inside adapter>
cd $HOME/ready
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
#type `exit` in the terminal to exit
```
