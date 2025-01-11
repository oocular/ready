# APIS

## Launch dev container
```
cd $HOME/repositories/ready/docs/holoscan
bash launch_dev_container.bash
```

## READY
* Download videos (`*.gxf_entities`, `*.gxf_index`, `*.mp4`) and models (`*.sim-BHWC.onnx`) from this [shared directory](https://liveuclac-my.sharepoint.com/:f:/r/personal/ccaemxo_ucl_ac_uk/Documents/nvidia-clara-agx/READY-Hawkes-Hackathon-2024/models-and-videos?csf=1&web=1&e=7uNpQH).

* Run app using your own repository in `dev_container` of clara-agx
```
cd /workspace/volumes/ready/scripts/apis
bash ready_py.bash replayer #or v4l2
```

* Debug application in local host device
```
cd $HOME/repositories/ready/src/ready/apis/holoscan/ready/python
vim -O ready.py ready-mobious.yaml ##Ctrl+WW to swap windows; :vertical resize 100
```

* TODO section!
```
#TODO: add model,path,name to config files and refactor script to pass such values to avoid long arguments inputs!
#TODO: The following section is confusing. Let's clean it. Models were tested but need to be cleaned or perhaps just remove them!
## novel
clear && python ready.py -d /workspace/volumes/ready/data/novel -m model3ch-23jul2024t0716-sim-BHWC.onnx -l logger.log -df TRUE
## openEDS
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024.onnx -l logger.log -df TRUE
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024-sim.onnx -l logger.log -df TRUE
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024-sim-BHWC.onnx -l logger.log -df TRUE
## openEDS: model3ch-23jul2024t0716-sim-BHWC
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m model3ch-23jul2024t0716-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
clear && python ready.py -d /workspace/volumes/ready/data/openEDS -m _weights_10-09-24_23-53-45-sim-BHWC.onnx -l logger.log -df TRUE -s replayer #v4l2
## mobious
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_02-09-24_21-02-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_02-09-24_22-24_trained10e_8batch_1143trainset-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_03-09-24_19-16-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_04-09-24_16-31-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_10-09-24_03-46-29-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
clear && python ready.py -d /workspace/volumes/ready/data/mobious -m _weights_10-09-24_04-50-40-sim-BHWC.onnx -l logger.log -df TRUE -s replayer
```

## v4l2_camera

* On dev container
```
cd /workspace/volumes/ready/scripts/apis #cd /workspace/volumes/ready/src/ready/apis/holoscan/v4l2_camera/python
bash v4l2_cam.bash
```

* On local device host
```
cd $HOME/repositories/ready/src/ready/apis/holoscan/v4l2_camera/python/
vim -O v4l2_camera.py v4l2_camera.yaml
```

## Bring Your Own Model
This example shows how to run inference with Holoscan and provides a mechanism, to replace the existing identity model with another model.
* Identity ONNX model at `model/identity_model.onnx`

![fig](../figs/identity_model_onnx_netronapp.png)

* Run app using your own repository in dev_container of clara-agx
```
cd /workspace/volumes/ready/src/ready/apis/holoscan/bring_your_own_model/python
python byom.py -d /workspace/volumes/ready/data/openEDS -m identity_model.onnx -l logger.log
#python byom.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024.onnx -l logger.log
#python byom.py -d /workspace/volumes/ready/data/openEDS -m model-5jul2024-sim.onnx -l logger.log
```

* Debug application in local host device
```
cd $HOME/repositories/ready/src/ready/apis/holoscan/bring_your_own_model/python
vim -O byom.py byom.yaml ##Ctrl+WW to swap windows; :vertical resize 100
```

## WebRTC Video Client
* Launching `webrtc_video_client`
```
cd /workspace/volumes/ready/scripts/apis
bash webrtc.bash LOCAL #PUBLIC
```

* Open browser
```
firefox http://127.0.0.1:8080/
```

* Conneting from a different machine
```
export PYTHONPATH=${PYTHONPATH}:/workspace/holohub
cd /workspace/volumes/ready/src/ready/apis/holoscan/webrtc
openssl req -new -newkey rsa:4096 -x509 -sha256 -days 365 -nodes -out MyCertificate.crt -keyout MyKey.key #just pressed enter
python webrtc_client.py --cert-file MyCertificate.crt --key-file MyKey.key
```
* Go to `chrome://flags`, search for the flag `unsafely-treat-insecure-origin-as-secure`, enter the origin you want to treat as secure such as `http://{YOUR HOST IP}:8080`, enable the feature and relaunch the browser.
See further details [here](https://github.com/nvidia-holoscan/holohub/tree/main/applications/webrtc_video_client)

* video-resolution: 320x240, 640x480, 960x540, 1280x720, 1920x1080
* video-codec: VP8, H264

## References
* Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) for step-by-step documentation of this example.
* https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/bring_your_own_model 
* https://github.com/nvidia-holoscan/holohub/tree/e1453b36a652682865d6d9d807d565435ca4f16f/applications/ssd_detection_endoscopy_tools
