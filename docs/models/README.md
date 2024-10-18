# Models

## Training models
### Debug models local device
* activate and export libs
```
#cd root path of repo
source .venv/bin/activate #To activate the virtual environment:
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```
### Train models in server
* train/debug unet with openEDS 
```
#train
python src/ready/apis/train.py

#debug model
cd src/ready
vim -O apis/train.py utils/datasets.py
```

* train/debug unet with mobious
```
#train
python src/ready/apis/train_mobious.py

#debug model
cd src/ready
vim -O apis/train_mobious.py utils/datasets.py
```

## Copying model to local host
```
#openEDS
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS/models/* ~/Desktop/nystagmus-tracking/datasets/openEDS/models/trained_models_in_cricket
#100%   89MB   6.2MB/s   00:14 
#100%  268MB   6.2MB/s   00:43

#MOBIOUS
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious/MOBIOUS/models/* ~/Desktop/nystagmus-tracking/datasets/mobious/models/trained_models_in_cricket
#100%   89MB   3.8MB/s   00:23 # at 44.7KB/s 34:06
```

## Optimise model
Go to [data](../../data/) and respective models path 
