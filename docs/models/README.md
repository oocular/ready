# Models

## Debug and train models in local device
Run and/or edit bash scripts [train](../../scripts/models/train_unet_with_mobious.bash) and [config](../../configs/models/unet/config_train_unet_with_mobious.yaml) in the terminal.
```
bash scripts/models/train_unet_with_mobious.bash
vim configs/models/unet/config_train_unet_with_mobious.yaml
vim src/ready/apis/train_mobious.py
```

## Train models in server
We recommend to see [README](../cricket/README.md).

## untar models in local device
```
cd ~/datasets/ready/mobious/trained_models_in_cricket
tar -xvzf weights_29-Mar-2025_16-23-29_with_augmenations_train100per_1144_trained_in_30139s.tar.gz
├── [2.1K]  loss_values_29-Mar-2025_16-23-29.csv
├── [ 236]  performance_29-Mar-2025_16-23-29.json
└── [ 89M]  _weights_29-Mar-2025_16-23-29.pth
```

## Optimise models
Go to [data](../../data/) and respective models path 
