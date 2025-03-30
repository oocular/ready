# Models with augmentations


## Model path and files
```
~/datasets/ready/mobious/trained_models_in_cricket/29-Mar-2025_16-23-29$ tree -h
[4.0K]  .
├── [2.1K]  loss_values_29-Mar-2025_16-23-29.csv
├── [ 236]  performance_29-Mar-2025_16-23-29.json
└── [ 89M]  _weights_29-Mar-2025_16-23-29.pth
```

## Preparations
### Conversion to ONNX (using .pth models) and ONNX symplification 
```
cd $HOME/repositories/oocular/ready
#EDIT config
vim configs/models/unet/config_convert_to_onnx_and_simplify_it.yaml
#CONVERT model
bash scripts/models/convert_to_onnx_and_simplify_it.bash
```


## Rebinding model to new nodes (NCHW to NHWC)
```
cd $HOME/repositories/oocular/ready
bash scripts/models/rebing_model_NCWH_to_NHWC.bash
```


## Model properties with https://netron.app/

### Graph properties of `_weights_29-Mar-2025_16-23-29-sim-BHWC.onnx`
```
name
main_graph
INPUT__0
name: INPUT__0
tensor: float32[1,400,640,3]
output_old
name: output_old
tensor: float32[batch_size,4,400,640]
```



