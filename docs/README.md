# Documentation

## lauch container
```
bash docs/launch_container.bash
##inside adapter>
cd $HOME/ready
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
#type `exit` in the terminal to exit

```

## installing package in local machine (not need if you are using container)
```
conda create -n "readyVE" python=3.12 pip -c conda-forge
conda activate readyVE
pip install --editable . # Install the package in editable mode
pip install .[test]
pip install .[learning]
#pip uninstall ready
#conda deactivate
#conda remove -n readyVE --all
```

## Debugging
```
conda activate readyVE
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```

## Testing 
```
python -m pytest -v -s tests
python -m pytest -v -s tests/test_data_paths.py::test_txt
python -m pytest -v -s tests/test_data_paths.py::test_masks
```

## Learning pipeline
```
cd $HOME/ready/src/ready/models #change path
vim $HOME/ready/src/ready/models/unet.py #edit
python src/ready/models/train.py
python src/ready/apis/inference.py
```

## Pre-commit
```
pre-commit run -a
```

## References

### Code

https://github.com/vital-ultrasound/ai-echocardiography-for-low-resource-countries/tree/main/scripts/learning-pipeline     
https://github.com/vital-ultrasound/ai-echocardiography-for-low-resource-countries/blob/main/source/models/architectures.py   
https://distill.pub/2016/deconv-checkerboard/    
https://github.com/PRLAB21/MaxViT-UNet   

https://github.com/IMSY-DKFZ/htc  

### Datasets 

* https://heiporspectral.org/
	* https://figures.heiporspectral.org/view_organs/01_stomach/P086%232021_04_15_11_38_26.html

### Literature 
Duvieusart, Benjamin, Terence S. Leung, Nehzat Koohi, and Diego Kaski. "Digital biomarkers from gaze tests for classification of central and peripheral lesions in acute vestibular syndrome." Frontiers in N    eurology 15 (2024): 1354041. https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1354041/full

### Blogs
* Multi-target in Albumentations
	* https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e 






