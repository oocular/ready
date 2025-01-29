# Data

The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. 
The total volume of the dataset is 72 images grouped into 6 larger tiles. 

The classes are:
Building: #3C1098  
Land (unpaved area): #8429F6  
Road: #6EC1E4  
Vegetation: #FEDD3A  
Water: #E2A929  
Unlabeled: #9B9B9B  

The images were segmented by the trainees of the Roia Foundation in Syria.   


## Download 
Download dataset from https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery?resource=download


## References 
https://github.com/ayushdabra/dubai-satellite-imagery-segmentation    


## Dataset
```
$ tree -s
[       4096]  .
├── [       4096]  models
│   ├── [         75]  losses_07.233587.cvs
│   └── [  405692246]  unetvit_epoch_0_7.23359.pth
├── [       4096]  tile01
│   ├── [       4096]  images
│   │   ├── [     147167]  image_part_001.jpg
│   │   ├── [     136801]  image_part_002.jpg
│   │   ├── [     146019]  image_part_003.jpg
│   │   ├── [     139488]  image_part_004.jpg
│   │   ├── [     131770]  image_part_005.jpg
│   │   ├── [     139988]  image_part_006.jpg
│   │   ├── [     146863]  image_part_007.jpg
│   │   ├── [     161169]  image_part_008.jpg
│   │   └── [     161406]  image_part_009.jpg
│   └── [       4096]  masks
│       ├── [      28987]  image_part_001.png
│       ├── [      23215]  image_part_002.png
│       ├── [      57459]  image_part_003.png
│       ├── [      34217]  image_part_004.png
│       ├── [      28722]  image_part_005.png
│       ├── [      55608]  image_part_006.png
│       ├── [      35312]  image_part_007.png
│       ├── [      59483]  image_part_008.png
│       └── [      65160]  image_part_009.png
└── [       4096]  tile02
    ├── [       4096]  images
    │   ├── [      98806]  image_part_001.jpg
    │   ├── [      89266]  image_part_002.jpg
    │   ├── [      81855]  image_part_003.jpg
    │   ├── [      79977]  image_part_004.jpg
    │   ├── [      74545]  image_part_005.jpg
    │   ├── [      85626]  image_part_006.jpg
    │   ├── [      84234]  image_part_007.jpg
    │   ├── [      86118]  image_part_008.jpg
    │   └── [      85465]  image_part_009.jpg
    └── [       4096]  masks
        ├── [      14244]  image_part_001.png
        ├── [       9923]  image_part_002.png
        ├── [       6167]  image_part_003.png
        ├── [       6434]  image_part_004.png
        ├── [       9271]  image_part_005.png
        ├── [       5912]  image_part_006.png
        ├── [       9188]  image_part_007.png
        ├── [       5411]  image_part_008.png
        └── [       5251]  image_part_009.png

7 directories, 38 files
#24 directories, 145 files #original dataset size
```

