# Dependencies

## Creating virtual environment for python
* Using conda
```
conda create -n "readyVE" python=3.12 pip -c conda-forge
conda activate *VE
conda list -n *VE #to check installed packages
conda remove -n *VE --all #in case you want to remove it
```

* Using python virtual environment
```
# Installing dependencies in Ubuntu 22.04
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv
# Create path for venv
cd $HOME
mkdir *VE
cd *VE
# Create virtual environment
python3 -m venv *VE
source *VE/bin/activate
```

* Create pylintc
```
cd tests/
pylint --generate-rcfile > .pylintrc
```
