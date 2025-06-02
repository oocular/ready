# Demos

`READY` demo aplication ([ready.py](/src/ready/apis/holoscan/ready/python/ready.py)) is running in a local host LaptopGPU with NVIDIARTXA2000-8GB using local-built holoscan-sdk. 
[UNet](src/ready/models/unet.py) was trained in cricket with A100-40GB  and using [27.4K images of 1 channel](data/openEDS/README.md). 

| Animation | Data, Model(s), API |
| --- | --- |
| ![fig](../../docs/figs/animations/ready-demo-2024-07-24_07.52.36-ezgif.com-video-to-gif-converter.gif) **Fig.** Initial demo using UNET with OpenEDS datsets. Video has 3 frames copied 10 times to create a 30 frame per second video. | [DATA: OpenEDS](../data/openEDS);  [MODEL: UNET](../data/openEDS/models); [API: ready.py d98c497](https://github.com/UCL/ready/blob/d98c497392ba7d91e9218fa5b73c75c629e3d29b/src/ready/apis/holoscan/ready/python/ready.py)
| ![fig](../../docs/figs/animations/ready-demo-2024-08-11_18.29.22-ezgif.com-video-to-gif-converter.gif) **Fig.** Pupil segmentation was masked to compute its centroid values as coordinates. Centroid values for x are plotted as cummulative time series of 30 samples. Streamed video consist of different 6 frames copied 5 times to create a 30 frames per second video. | [DATA: OpenEDS](../data/openEDS); [MODEL: UNET](../data/openEDS/models);  [API: ready.py b44515a](https://github.com/UCL/ready/blob/b44515a70727620187f20ea19c50c77f4cacbad6/src/ready/apis/holoscan/ready/python/ready.py)  |

