# Models

## Local path for models and size
```
tree -h
[4.0K]  .
└── [ 89M]  _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset.pth
```

## Inference TODO!!

## Preparations
### Conversion to ONNX
```
conda activate readyVE
cd $HOME/src
export PYTHONPATH=.
python src/ready/apis/convert_to_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/mobious/weights/trained_models_in_cricket -i _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset.pth
```

### ONNX symplification
```
python src/ready/apis/sim_onnx.py -p $HOME/Desktop/nystagmus-tracking/datasets/mobious/weights/trained_models_in_cricket -m _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset.onnx
```


## Rebinding model to new nodes (NCHW to NHWC)
```
conda activate readyVE
cd ~/ready/data/openEDS/models
#copy simplified models to  ~/ready/data/openEDS/models
cp $HOME/Desktop/nystagmus-tracking/datasets/mobious/weights/trained_models_in_cricket/_weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim.onnx .
pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

python ../../../src/ready/apis/holoscan/utils/graph_surgeon.py _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim.onnx _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim-BHWC.onnx 3 400 640
```


## Properties with https://netron.app/

* `_weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset.onnx`

```
format: ONNX v8
producer: pytorch 2.3.1
version: 0
imports: ai.onnx v16
graph: main_graph


name: input
tensor: float32[batch_size,3,400,640]
name: output
tensor: float32[batch_size,4,400,640]
```

* `_weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim.onnx`
```
name: input
tensor: float32[batch_size,3,400,640]
name: output
tensor: float32[batch_size,4,400,640]
```


* `_weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim-BHWC.onnx`

```

format ONNX v10
producer pytorch 2.3.1
version 0 
imports ai.onnx v16
graph main_graph

name: INPUT__0
tensor: float32[1,400,640,3]
output_old
name: output_old
tensor: float32[batch_size,4,400,640]
```


