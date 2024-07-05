# “cricket” – ARM CPU, Nvidia GPU

## features
* HD: 5TB
* GPU: NVIDIA A100, A100 , 40 GB HBM2, 1.6 TB/sec https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
* https://www.ucl.ac.uk/advanced-research-computing/coming-soon-platforms

## connections

1. connect to `vpn.ucl.ac.uk` using cisco https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
2. connect to the server
```
ssh ccxxxxx@cricket.rc.ucl.ac.uk
xterm -rv & # open as many terminals you want
```

## Copying dataset
scp openEDS.zip ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS #openEDS.zip #8.0G ETA 1h at 2MB/s


## /opt/nvidia/containers (first time)
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



