# APIS

## Launch dev container
```
cd $HOME/repositories/oocular/ready/docs/holoscan
bash launch_dev_container.bash
```

## READY
* Download videos (`*.gxf_entities`, `*.gxf_index`, `*.mp4`) and models (`*.sim-BHWC.onnx`) from this [shared directory](https://liveuclac-my.sharepoint.com/my?id=%2Fpersonal%2Fccaemxo%5Fucl%5Fac%5Fuk%2FDocuments%2FProjects%2FREADY%2Dshared%2Fready%2Fmodels%2Dand%2Dvideos%2Fmobious%2Fholoscan).

* Run app using your own repository in `dev_container` of clara-agx
```
cd /workspace/volumes/ready/scripts/apis
bash ready_py.bash replayer #or v4l2
```

* Debug application in local host device
```
cd $HOME/repositories/oocular/ready/src/ready/apis/holoscan/ready/python
vim -O ready.py ready-mobious.yaml ##Ctrl+WW to swap windows; :vertical resize 100
```

* TODO section!
```
#TODO: add model,path,name to config files and refactor script to pass such values to avoid long arguments inputs!
#TODO: The following section is confusing. Let's clean it. Models were tested but need to be cleaned or perhaps just remove them!
## novel
clear && python ready.py -d /workspace/volumes/ready/data/novel -m model3ch-23jul2024t0716-sim-BHWC.onnx -l logger.log -df TRUE
## openEDS
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024.onnx -l logger.log -df TRUE
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024-sim.onnx -l logger.log -df TRUE
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024-sim-BHWC.onnx -l logger.log -df TRUE
## openEDS: model3ch-23jul2024t0716-sim-BHWC
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model3ch-23jul2024t0716-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m _weights_10-09-24_23-53-45-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
## mobious
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_02-09-24_21-02-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_02-09-24_22-24_trained10e_8batch_1143trainset-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_03-09-24_19-16-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_04-09-24_16-31-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_10-09-24_03-46-29-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_10-09-24_04-50-40-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
```


