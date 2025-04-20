# Models with augmentations


## Untar Models

### weights_28-Mar-2025_none_augmentations.tar.gz
tar -xzf weights_28-Mar-2025_none_augmentations.tar.gz

### weights_29-Mar-2025_16-23-29_with_augmenations_train100per_1144_trained_in_30139s.tar.gz
tar -xzf weights_29-Mar-2025_16-23-29_with_augmenations_train100per_1144_trained_in_30139s.tar.gz

### weights_30-Mar-2025_15-14-24_with_augmenations_train050per_0572_trained_in_15030s.tar.gz
tar -xzf weights_30-Mar-2025_15-14-24_with_augmenations_train050per_0572_trained_in_15030s.tar.gz

### weights_29-Mar-2025_22-12-17_with_augmenations_train025per_0286_trained_in_7555s.tar.gz
tar -xzf weights_29-Mar-2025_22-12-17_with_augmenations_train025per_0286_trained_in_7555s.tar.gz

### weights_30-Mar-2025_08-35-44_with_augmenations_train012per_0145_trained_in_3778s.tar.gz
tar -xzf weights_30-Mar-2025_08-35-44_with_augmenations_train012per_0145_trained_in_3778s.tar.gz


## Plot performance of models
* Files
```
28-Mar-2025_15-25-07
    loss_values_28-Mar-2025_15-25-07.csv
    performance_28-Mar-2025_15-25-07.json

29-Mar-2025_16-23-29
    loss_values_29-Mar-2025_16-23-29.csv
    performance_29-Mar-2025_16-23-29.json


30-Mar-2025_15-14-24
    loss_values_30-Mar-2025_15-14-24.csv
    performance_30-Mar-2025_15-14-24.json


29-Mar-2025_22-12-17
    loss_values_29-Mar-2025_22-12-17.csv
    performance_29-Mar-2025_22-12-17.json


30-Mar-2025_08-35-44
    loss_values_30-Mar-2025_08-35-44.csv
    performance_30-Mar-2025_08-35-44.json
```
* Plotting
```
cd $HOME/repositories/oocular/ready
source .venv/bin/activate
python src/ready/apis/plot_losses.py -c configs/apis/plot_losses.yml
python src/ready/apis/plot_performance.py -c configs/apis/plot_performance.yml
```

## Model optimisitation
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

### Graph properties of `_weights_29-Mar-2025_16-23-29-sim-BHWC.onnx`; `_weights_28-Mar-2025_15-25-07-sim-BHWC.onnx`
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
