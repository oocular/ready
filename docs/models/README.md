# Models

## Training models
### Debug models local device
* activate and export libs
```
#cd root path of repo
source .venv/bin/activate #To activate the virtual environment:
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```
* Prototyping unetvit
```
# test dataset
pytest -vs tests/test_unetvit.py::test_segDataset
# train, optimise and test inference
python src/ready/apis/train_unetvit.py
python src/ready/apis/pytorch2onnx.py -i <model_name>.pth
pytest -vs tests/test_unetvit.py::test_inference
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
python src/ready/apis/train_mobious.py -df #0or1

#debug model
cd src/ready
vim -O apis/train_mobious.py utils/datasets.py
```

### UNetVIT changes
```
To successfully run code with old data:

1. First, add pytest configuration to your pyproject.toml. Add this section:
[tool.pytest]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"
pythonpath = [
    "src"
]
2. Make sure you have all the necessary __init__.py files in your directory structure:
touch src/__init__.py
touch src/ready/__init__.py
touch src/ready/models/__init__.py

3. Modify your import statement in tests/test_unetvit.py to be:
from ready.models.unetvit import UNetViT

4. Clean up any existing installation and reinstall:
pip uninstall ready -y
pip install -e ".[test,learning]"



Dataset Class Changes:


Added new MOBIUSDataset class to handle different directory structure
Original: Used segDataset with tile01/tile02 structure
New: Handles numbered folders (1-35) in Images/Masks directories
Added explicit mask normalization (0/255 → 0/1)


Directory Structure Handling:
Original:
DATASET_PATH = "/data/test-samples/semantic-segmentation-aerial-imagery"

tile01/images, tile01/masks, tile02/images, tile02/masks
New:
DATASET_PATH = "/data/test-samples/MOBIUS"
Images/1..35, Masks/1..35

Image/Mask Processing:
Original:
dataset = segDataset(DATASET_PATH, training=True, transform=t)

New:
image = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path)
mask = np.array(mask)
if len(mask.shape) == 3:
mask = mask[:, :, 0]  # Handle multi-channel masks
mask = (mask > 0).astype(np.int64)  # Normalize to binary 0/1

Transform Changes:
Original:
t = transforms.Compose([color_shift, blurriness])

New:
resize = transforms.Resize((512, 512))  # Added resize for varying sizes
t = transforms.Compose([resize, color_shift, blurriness])

Data Loading:


Original: Assumed consistent image sizes (512x512)
New: Added resize transform to handle varying image sizes
Added better error handling and validation for image/mask pairs


Debug Information:
Added more detailed logging:
logger.info(f"Mask unique values: {torch.unique(single_set[1])}")
logger.info(f"Mask dtype: {single_set[1].dtype}")
Removed Fixed Assertions:
Original:
assert len(test_dataloader) == 1
assert len(train_dataloader) == 5

New:
logger.info(f"Number of test batches: {len(test_dataloader)}")
logger.info(f"Number of train batches: {len(train_dataloader)}")
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
