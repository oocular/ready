# :robot: :eye: READY: REal-time Ai Diagnosis for nYstagmus 	
This repository contains documentation and code for `REal-time Ai Diagnosis for nYstagmus` project.

## installing package
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

## Pre-commit
```
pre-commit run -a
```

## Testing 
```
# dataset
#export PYTHONPATH=$HOME/<ADD_REPO_PATH> #$PWD
python -m pytest -v -s tests
python -m pytest -v -s tests/test_data_paths.py::test_txt
python -m pytest -v -s tests/test_data_paths.py::test_masks
```

## Learning pipeline
```
cd ready/src/ready/openeds
python src/ready/openeds/segnet.py
```

## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Clone the repository by typing (or copying) the following lines in a terminal
```
git clone git@github.com:UCL/ready.git
```

