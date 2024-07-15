# openEDS
> OpenEDS (Open Eye Dataset) is a large scale data set of eye-images captured using a virtual-reality (VR) head mounted display mounted with two synchronized eyefacing cameras at a frame rate of 200 Hz under controlled illumination.
> This dataset is compiled from video capture of the eye-region collected from 152 individual participants and is divided into four subsets: 
	(i) 12,759 images with pixel-level annotations for key eye-regions: iris, pupil and sclera 
	(ii) 252,690 unlabelled eye-images, 
	(iii) 91,200 frames from randomly selected video sequence of 1.5 seconds in duration and 
	(iv) 143 pairs of left and right point cloud data compiled from corneal topography of eye regions collected from a subset, 143 out of 152, participants in the study.

https://www.kaggle.com/datasets/soumicksarker/openeds-dataset/data

## Local datasets path
cd $HOME/datasets/openEDS

```
train/
	images #27,431 #640 x 400 pixels #153.3 kB #PNG image
	labels #27,431 #(400, 640) #*npy
	masks  #27,431 #640 x 400 pixels #1.6 kB #PNG image

test/
	images #2,744
	labels #2,744
	masks #2,744

validation/
	images #2,744
	labels #2,744
	masks #2,744
```

## Previewing data
The following images is created using [inference.py](../../src/ready/apis/inference.py)
![figs](../../docs/figs/openEDS-dataset-models.svg)


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

* https://netron.app/

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

![figs](../../docs/figs/models-at-neutronapp.svg)





