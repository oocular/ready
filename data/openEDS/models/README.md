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

## Inference 
* Inference for three frames
![figs](../../../docs/figs/inference-val3frames.svg)


## Preparations
* Conversion to ONNX
```
conda activate readyVE
cd $HOME/src
export PYTHONPATH=.
python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -i model-5jul2024.pth
python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -i model3ch-23jul2024t0716.pth
```

* ONNX symplification
```
python src/ready/apis/sim_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -m model-5jul2024.onnx
python src/ready/apis/sim_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/openEDS/weights/trained_models_in_cricket -m model3ch-23jul2024t0716.onnx
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
cd ~/ready/data/openEDS/models

pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
python ../../../src/ready/apis/holoscan/utils/graph_surgeon.py model-5jul2024-sim.onnx model-5jul2024-sim-BHWC.onnx 1 400 640
python ../../../src/ready/apis/holoscan/utils/graph_surgeon.py model3ch-23jul2024t0716-sim.onnx model3ch-23jul2024t0716-sim-BHWC.onnx 3 400 640
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


* `model3ch-23jul2024t0716-sim.onnx` at https://netron.app/
```
input
name: input
tensor: float32[batch_size,3,400,640]
output
name: output
tensor: float32[batch_size,4,400,640]
```

* `model3ch-23jul2024t0716-sim-BHWC.onnx` https://netron.app/
```
INPUT__0
name: INPUT__0
tensor: float32[1,400,640,3]
output_old
name: output_old
tensor: float32[batch_size,4,400,640]
```

## Real-time Inference with holoscan-sdk

* Holocan-sdk api (val-000160-640wX400h)

![figs](../../../docs/figs/holoscan-api-video_30_duplicated_frames_of_val-000160-640wX400h_rgb_channels3.png)


* Four images scaled to 640wX400h

![fig](../../../docs/figs/holoscan-sdk-four-frames-in-one.png)

* Fractal images in a scaled resolution of 640wX400h

![fig](../../../docs/figs/holoscan-sdk-fractal-frames-in-one.png)

