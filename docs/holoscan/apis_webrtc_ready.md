# WebRTC Video Client [:link:](https://github.com/nvidia-holoscan/holohub/tree/main/applications/webrtc_video_client)

## Create certificates to connect from a different machine

```bash
cd $HOME/datasets/ready
mkdir -p webrtc && cd webrtc
openssl req -new -newkey rsa:4096 -x509 -sha256 -days 365 -nodes -out MyCertificate.crt -keyout MyKey.key 
#JUST PRESS ENTER TO USE DEFAULT VALUES, e.g.
# .....+...++++++
# -----
# You are about to be asked to enter information that will be incorporated
# into your certificate request.
# What you are about to enter is what is called a Distinguished Name or a DN.
# There are quite a few fields but you can leave some blank
# For some fields there will be a default value,
# If you enter '.', the field will be left blank.
# -----
# Country Name (2 letter code) [AU]:
# State or Province Name (full name) [Some-State]:
# Locality Name (eg, city) []:
# Organization Name (eg, company) [Internet Widgits Pty Ltd]:
# Organizational Unit Name (eg, section) []:
# Common Name (e.g. server FQDN or YOUR name) []:
# Email Address []:
```

## Edit configuration file to setup model and recorderd path and filenames
```bash
cd $HOME/repositories/oocular/ready/
vim configs/apis/config_webrtc_ready.yaml
```

## Launch dev container
```bash
cd $HOME/repositories/oocular/ready/docs/holoscan
bash launch_dev_container.bash
```

## Run application on `PUBLIC` network

* Launching `webrtc_client`
```bash
#RECORDER FALSE
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash PUBLIC DEGUG webrtc False
#RECORDER TRUE
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash PUBLIC DEGUG webrtc True
```

* Check your host IP

Using `ifconfig` command in your terminal you can see a log like the following where the `000.000.0.000` is your `{YOUR_HOST_IP}`.
```bash
$ifconfig
wlp0s20f3: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 000.000.0.000 netmask 255.255.255.0  broadcast 0.0.0.0
```

* Open a browser in your mobile phone or any device connect to the network
If using `chrome`, go to `chrome://flags`, search for the flag `unsafely-treat-insecure-origin-as-secure`, enter the origin you want to treat as secure such as `http://${YOUR_HOST_IP}:8080`, enable the feature and relaunch the browser. 
See further details [here](https://github.com/nvidia-holoscan/holohub/tree/main/applications/webrtc_video_client).


* Stop pipeline
    * CTRL-D in the bash and then Stop and exit api

* Replay recordings
```bash
#REPLAYER replayer_raw
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash PUBLIC DEGUG replayer_raw False
#REPLAYER replayer_inference
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash PUBLIC DEGUG replayer_inference False
```


## Run application on `LOCAL` network

* Launching `webrtc_client`
```bash
#RECORDER FALSE
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash LOCAL DEGUG webrtc False
#RECORDER TRUE
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash LOCAL DEGUG webrtc True
```

* Open browser on local network
```bash
firefox http://127.0.0.1:8080/
```

* video-resolution: 320x240, 640x480, 960x540, 1280x720, 1920x1080
* video-codec: VP8, H264

![fig](../figs/webrtc_app.png)


* Stop and exit api
    * Stop streaming
    * Close api

* Replay recordings
```bash
#REPLAYER replayer_raw
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash LOCAL DEGUG replayer_raw False
#REPLAYER replayer_inference
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash LOCAL DEGUG replayer_inference False
```

## Useful commands

* USAGE of webrtc_ready.bash
```bash
bash webrtc_ready.bash <$1:NET: LOCAL/PUBLIC> <$2:HOLOSCAN_LOG_LEVEL: OFF/DEBUG/TRACE/INFO/ERROR> <$3:SOURCE: webrtc/replayer> <$4:ENABLE_RECORDING: True/False>
```


## Graph structure for [webrtc_client.py](../../src/ready/apis/holoscan/webrtc/webrtc_client.py)
```mermaid
flowchart LR
    subgraph Server
        WebRTCClientOp --> DropFramesOp
        DropFramesOp --> HolovizOp
        DropFramesOp --> PreInfoOp
        PreInfoOp --> FormatOp
        FormatOp --> InferenceOp
        InferenceOp --> SegmentationOp
        SegmentationOp --> HolovizOp
        InferenceOp --> PostInfoOp
        PostInfoOp --> HolovizOp_outputs --> HolovizOp
        PostInfoOp --> HolovizOp_output_specs --> HolovizOp
        WebServer
    end
    subgraph Client
        Webcam --> Browser
        Browser <--> WebRTCClientOp
        Browser <--> WebServer
    end
```

See more [flow_benchmarking]( ../../data/webrtc/flow_benchmarking/)

## Troubleshouting
```bash
#KILL script
kill $(ps aux | grep "python webrtc_client.py" | awk '{print $2}')

#EDIT SCRIPTS LOCALLY
cd $HOME/repositories/oocular/ready/
vim scripts/apis/webrtc_ready.bash
vim src/ready/apis/holoscan/webrtc_ready/webrtc_client.py

## STOP dev container
docker stop $(docker ps -q | head -n 1) # choose the dev container ID from `docker ps` or get ids by using `$(docker ps -aq)`
```
