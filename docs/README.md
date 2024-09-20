# Documentation

## Install uv (An extremely fast Python package manager)
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create venv
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment:
deactivate

#uv venv 2nd_env --python 3.13 #create with a diff python version
#rm -rf 2nd_env #to remove 2nd_env
```

## Install python package deps
```
uv pip install --editable . # Install the package in editable mode
uv pip install .[test]
uv pip install .[learning]
#uv pip uninstall ready
```

## Debugging
```
source .venv/bin/activate #To activate the virtual environment:
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```

## Testing 
```
python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset
python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset_labels

#TODO: Fix the following tests with data in the repo
python -m pytest -v -s tests/test_data_paths.py::test_txt
python -m pytest -v -s tests/test_data_paths.py::test_png
python -m pytest -v -s tests/test_data_paths.py::test_masks
python -m pytest -v -s tests/test_data_paths.py::test_tif_with_matplotlib
```


## Pre-commit
```
#TODO run pre-commit with no errors!
pre-commit run -a
```


## Demos
`READY` demo aplication ([ready.py](/src/ready/apis/holoscan/ready/python/ready.py)) is running in a local host LaptopGPU with NVIDIARTXA2000-8GB using local-built holoscan-sdk. 
[UNet](src/ready/models/unet.py) was trained in cricket with A100-40GB  and using [27.4K images of 1 channel](data/openEDS/README.md). 

| Animation | Data, Model(s), API |
| --- | --- |
| ![fig](../docs/figs/animations/ready-demo-2024-07-24_07.52.36-ezgif.com-video-to-gif-converter.gif) **Fig.** Initial demo using UNET with OpenEDS datsets. Video has 3 frames copied 10 times to create a 30 frame per second video. | [DATA: OpenEDS](../data/openEDS);  [MODEL: UNET](../data/openEDS/models); [API: ready.py d98c497](https://github.com/UCL/ready/blob/d98c497392ba7d91e9218fa5b73c75c629e3d29b/src/ready/apis/holoscan/ready/python/ready.py)
| ![fig](../docs/figs/animations/ready-demo-2024-08-11_18.29.22-ezgif.com-video-to-gif-converter.gif) **Fig.** Pupil segmentation was masked to compute its centroid values as coordinates. Centroid values for x are plotted as cummulative time series of 30 samples. Streamed video consist of different 6 frames copied 5 times to create a 30 frames per second video. | [DATA: OpenEDS](../data/openEDS); [MODEL: UNET](../data/openEDS/models);  [API: ready.py b44515a](https://github.com/UCL/ready/blob/b44515a70727620187f20ea19c50c77f4cacbad6/src/ready/apis/holoscan/ready/python/ready.py)  |


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

