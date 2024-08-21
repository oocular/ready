# Models

## Training models
* activate and export libs
```
#debug models
conda activate readyVE
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```
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



## Inference in local device
```
cd $HOME_REPO
conda activate readyVE
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>

#inference
python src/ready/apis/inference.py
vim src/ready/apis/inference.py
```

