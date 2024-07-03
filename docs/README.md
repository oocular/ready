# docs

## lauch container
```
bash docs/launch_container.bash
#inside adapter>
cd $HOME/ready
export PYTHONPATH=$PWD #$HOME/<ADD_REPO_PATH>
```

## installing package (not need if you are using container)
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

## Testing 
```
python -m pytest -v -s tests
python -m pytest -v -s tests/test_data_paths.py::test_txt
python -m pytest -v -s tests/test_data_paths.py::test_masks
```

## Learning pipeline
```
cd $HOME/ready/src/ready/openeds #change path
vim $HOME/ready/src/ready/openeds/segnet.py #edit
python $HOME/ready/src/ready/openeds/segnet.py #run
```

## Pre-commit
```
pre-commit run -a
```

## References
* https://github.com/vital-ultrasound/ai-echocardiography-for-low-resource-countries/tree/main/scripts/learning-pipeline
* Duvieusart, Benjamin, Terence S. Leung, Nehzat Koohi, and Diego Kaski. "Digital biomarkers from gaze tests for classification of central and peripheral lesions in acute vestibular syndrome." Frontiers in N    eurology 15 (2024): 1354041. https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1354041/full

