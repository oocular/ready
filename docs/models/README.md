# Models


## Training models
```
#debug models
cd $HOME/ready/src/ready/models #change path
vim $HOME/ready/src/ready/models/unet.py #edit

#train
python src/ready/models/train.py

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

