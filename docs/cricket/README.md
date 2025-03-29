# Cricket machine: ARM CPU with A100 Nvidia GPU (80GB)

## Features
* HD: 5TB
* GPU: NVIDIA A100, A100 , 80 GB HBM2, 1.6 TB/sec https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
* https://www.ucl.ac.uk/advanced-research-computing/coming-soon-platforms

## Connect
1. Connect to `vpn.ucl.ac.uk` using cisco https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
2. Connect to the server
```
ssh -X ccxxxxx@cricket.rc.ucl.ac.uk
xterm -rv & # open as many terminals you want
```

## Copying dataset from local device to server
```
# openEDS.zip
scp openEDS.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS #openEDS.zip #8.0GB ETA 1h at 2MB/s
# MOBIUS.zip
scp MOBIUS.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious #MOBIUS.zip #3.3GB ETA  26mins at 2MB/s
scp strain-morbious.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious #34MB   1.9MB/s   00:17
```

## Copying files (models) to local host
```
#tar path
tar czf weights_29-Mar-2025_with_augmentations00.tar.gz 29-Mar-2025/
#moving paths
SERVER_DATAPATH=/home/ready/datasets/ready/mobious/models
TARFILE=weights_29-Mar-2025_with_augmentations00.tar.gz
TARFILE=weights_28-Mar-2025_none_augmentations.tar.gz
LOCAL_DATAPATH=/home/mxochicale/datasets/ready/mobious/trained_models_in_cricket
scp ccaemxo@cricket.rc.ucl.ac.uk:${SERVER_DATAPATH}/${TARFILE} ${LOCAL_DATAPATH}
## Example with zip
#zip -r models12-12-24.zip models/
#scp ccaemxo@cricket.rc.ucl.ac.uk:/home/ready/datasets/mobious/MOBIOUS/models/_weights_15-12-24_07-00-10_TRAINe100_GPUa100_80gb.zip ~/Desktop/nystagmus-tracking/datasets/mobious/models/trained_models_in_cricket
```

## Container

### Setting up /opt/nvidia/containers (Optional)
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
bash docs/cricket/launch_container_in_cricket.bash <ADD_USERNAME (eg. ccxxxxx)>

#inside adapter>

## Create data paths 
mkdir -p $HOME/datasets/ready/mobious/models

## Change to project path
cd $HOME/ready
export PYTHONPATH=$HOME/ready/src #. #$HOME/<ADD_REPO_PATH>

python -m pip install --upgrade pip
pip install loguru
pip install omegaconf

#?pip install -e ".[test,learning,model_optimisation]"

## GOTO models/README.md for instructions to train model in cricket
#type `exit` in the terminal to exit
```
