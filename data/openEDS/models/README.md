# Models

## Local path for models and size
```
cd $HOME/datasets/openEDS/weights
$ tree -h
.
├── [  89M]  ADD_MODEL_NAME_VAR.onnx
├── [  89M]  model.pth
└── [ 268M]  o.pth
```

## Transfer 
* Copying model to local host
```
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS/weights/* ~/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket
#100%   89MB   6.2MB/s   00:14 
#100%  268MB   6.2MB/s   00:43
```

## Preparations
* Conversion to ONNX
```
python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -i model-5jul2024.pth
```

* ONNX symplification
```
python src/ready/apis/sim_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -m model-5jul2024.onnx
```
OR https://convertmodel.com/#input=onnx&output=onnx

## Properties with https://netron.app/

* unet-model.onnx
```
format ONNX v8
producer pytorch 2.3.1
version 0
imports ai.onnx v16
graph main_graph

input
name: input
tensor: float32[batch_size,1,400,640]

output
name: output
tensor: float32[batch_size,4,400,640]
```

![figs](../../../docs/figs/models-at-neutronapp.svg)

* Identity model 

![figs](../../../docs/figs/identity_model_onnx_netronapp.png)

## Rebinding model to new nodes (NCHW to NHWC)




```
conda activate readyVE
pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
python ../../../src/ready/apis/holoscan/utils/graph_surgeon.py model-5jul2024-sim.onnx model-5jul2024-sim-BHWC.onnx 1 400 640
```



* `model-5jul2024-sim.onnx`


name: input
tensor: float32[batch_size,1,400,640]
output
name: output
tensor: float32[batch_size,4,400,640]



* `model-5jul2024-sim-BHWC.onnx`
```
INPUT__0
name: INPUT__0
tensor: float32[1,400,640,1]
output_old
name: output_old
tensor: float32[batch_size,4,400,640]
```

