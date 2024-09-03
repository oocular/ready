# Documentation

## Installing package in local machine (not need if you are using container)
```
conda create -n "readyVE" python=3.12 pip -c conda-forge
conda activate readyVE
#cd to home repository path (`cd ../`)
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
python -m pytest -v -s tests/test_data_paths.py::test_png
python -m pytest -v -s tests/test_data_paths.py::test_masks
python -m pytest -v -s tests/test_data_paths.py::test_tif_with_matplotlib
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

https://github.com/nizhenliang/RAUNet
https://github.com/CAMMA-public/ConvLSTM-Surgical-Tool-Tracker 


### Datasets 

* https://heiporspectral.org/
	* https://figures.heiporspectral.org/view_organs/01_stomach/P086%232021_04_15_11_38_26.html

* CATARACTS
	* https://ieee-dataport.org/open-access/cataracts
	* https://cataracts.grand-challenge.org/CaDIS/
	* https://cataracts-semantic-segmentation2020.grand-challenge.org/

### Literature 
Duvieusart, Benjamin, Terence S. Leung, Nehzat Koohi, and Diego Kaski. "Digital biomarkers from gaze tests for classification of central and peripheral lesions in acute vestibular syndrome." Frontiers in N    eurology 15 (2024): 1354041. https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1354041/full

### Blogs
* Multi-target in Albumentations
	* https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e 

### Challenges

* Justified Referral in AI Glaucoma Screening
	* https://justraigs.grand-challenge.org/justraigs/
	* https://zenodo.org/records/10035093
	* https://www.sciencedirect.com/science/article/pii/S2666914523000325 
	* https://github.com/DM2LL/JustRAIGS-IEEE-ISBI-2024

