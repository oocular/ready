# Models

## Local path for models and size
```
cd $HOME/datasets/openEDS/weights
$ tree -h
.
├── [ 112M]  model.pth
└── [ 337M]  o.pth
```

## Transfer 
* Copying model to local machine
```
scp ccxxxxx@cricket.rc.ucl.ac.uk:~/datasets/openEDS/weights/model.pth ~/Desktop/nystagmus-tracking/weights/trained_models_in_cricket
#100%  112MB   4.4MB/s   00:25
#100%  337MB   4.0MB/s   01:23 
```

## Preparations

* Conversion
```
python convert_to_onnx.py --model_path $HOME/... --input_model_name *.pth --output_model_name *.onnx
```
https://netron.app/

https://convertmodel.com/#input=onnx&output=onnx

