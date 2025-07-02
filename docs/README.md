# Documentation

## Getting started

Note: the following instructions were created for Linux/MacOS computers. If you are using a Windows machine, please install WSL2 (Windows Subsystem For Linux). Instructions are be found [here](https://learn.microsoft.com/en-us/windows/wsl/install). 

### Install [uv](https://github.com/astral-sh/uv): "An extremely fast Python package manager".
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create venv
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment:
deactivate

#uv venv 2nd_env --python 3.13 #create with a diff python version
#rm -rf 2nd_env #to remove 2nd_env
```

### Install python package deps
```
uv pip install --editable . # Install the package in editable mode
uv pip install .[test]
uv pip install .[learning]
uv pip install .[model_optimisation]
uv pip install -e ".[test,learning,model_optimisation]" # Install the package in editable mode
#uv pip uninstall ready
```

### Debugging
```
source .venv/bin/activate #To activate the virtual environment:
export PYTHONPATH=. #$HOME/ready #$HOME/<ADD_REPO_PATH>
```

#### If you have problems installing onnxsim

For WSL2 users, you may encounter an issue when running ```uv pip install -e ".[test,learning,model_optimisation]"```, stating that cmake cannot be found. From my experience, this is because your machine is unable to install the onnxsim package, found in model_optimisation. To fix this, run `sudo apt-get install cmake`. This will install cmake onto your machine. Afterwards, run ```sudo apt-get install python3.12-dev```. For cmake to work, it requires the development files that the python3.12-dev package provides.

### Testing
```
python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset
python -m pytest -v -s tests/test_data_paths.py::test_mobious_dataset_labels

#TODO: Fix the following tests with data in the repo
python -m pytest -v -s tests/test_data_paths.py::test_txt
python -m pytest -v -s tests/test_data_paths.py::test_png
python -m pytest -v -s tests/test_data_paths.py::test_masks
python -m pytest -v -s tests/test_data_paths.py::test_tif_with_matplotlib
```

### Pre-commit
```
pre-commit run -a
```

 


