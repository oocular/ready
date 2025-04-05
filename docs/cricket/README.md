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

# install package
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test,learning,model_optimisation]"
```

# Launch container 
bash docs/cricket/launch_container_in_cricket.bash <ADD_USERNAME (eg. ccxxxxx)>

#inside adapter>

## Create data paths 
mkdir -p $HOME/datasets/ready/mobious/models

## Change to project path
cd $HOME/ready
export PYTHONPATH=$HOME/ready/src #. #$HOME/<ADD_REPO_PATH>

## GOTO models/README.md for instructions to train model in cricket but you can try:
```
bash scripts/models/train_unet_with_mobious.bash
vim configs/models/unet/config_train_unet_with_mobious.yaml
```

#type `exit` in the terminal to exit
```

## Copying files (models) to local host
The following are examples that you can use with different variables.
```
## tar paths in server
PATHMODEL=30-Mar-2025_08-35-44
TRAINDATA=train012per_0145
TRAINTIMESEC=3778
TARMODEL=weights_${PATHMODEL}_with_augmenations_${TRAINDATA}_trained_in_${TRAINTIMESEC}s.tar.gz
tar czf ${TARMODEL} ${PATHMODEL}

## Moving path in local device
SERVER_DATAPATH=/home/ready/datasets/ready/mobious/models
LOCAL_DATAPATH=/home/mxochicale/datasets/ready/mobious/models_cricket
TARFILE=weights_29-Mar-2025_16-23-29_with_augmenations_train100per_1144_trained_in_30139s.tar.gz
scp ccxxxxx@cricket.rc.ucl.ac.uk:${SERVER_DATAPATH}/${TARFILE} ${LOCAL_DATAPATH}
```