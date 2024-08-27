# Models

## Training models
### Debug models local device
* activate and export libs
```
conda activate readyVE
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
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS/weights/* ~/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket
#100%   89MB   6.2MB/s   00:14 
#100%  268MB   6.2MB/s   00:43

#MOBIOUS
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/mobious/MOBIOUS/weights/* ~/Desktop/nystagmus-tracking/datasets/mobious/weights/trained_models_in_cricket
#100%   89MB   3.8MB/s   00:23
```

## Inference in local device
```
cd $HOME_REPO
conda activate readyVE
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>

#inference
python src/ready/apis/inference.py
vim src/ready/apis/inference.py
```

