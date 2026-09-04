# Proof of concept

## Generic steps

* Protocol document exists and is approved
* Consent
* Screening & setup
	* See [apis_webrtc_ready](../../docs/holoscan/apis_webrtc_ready.md)
* Primary position recordings

The following steps are needed if nystagmus not recorded in previous step

* Saccade test (not designed to eval nystagmus but allows to check nystagmus)
* Pursuit test (not designed to eval nystagmus but allows to check nystagmus)
* Positional test
* Throughout, cross-cutting requirements
* Validation

## Specific protocol 

### Launch the dev container (in case you start from LOCAL)

```bash
cd $HOME/repositories/oocular/ready/docs/holoscan
bash launch_dev_container.bash
```

### Create data path and edit config file

* Create `recordings` path otherwise you will get `: 'recorder_op' - Failed to open index_file_stream_ with error: GXF_FAILURE`
```bash
mkdir -p ~/datasets/ready/webrtc/recordings/ && cd ~/datasets/ready/webrtc/recordings
mkdir -p ID002 ID003 ID004 ID005 ID006
```

* Edit file
```bash
# Setup config file
CONFIG_YAML=config_webrtc_ready_poc_sep2026.yaml
vim configs/apis/${CONFIG_YAML}
```

### Launch the WebRTC client via `webrtc_ready.bash`

```bash
# Setup config file
CONFIG_YAML=config_webrtc_ready_poc_sep2026.yaml

# Recording disabled
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG webrtc False

# Recording enabled
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG webrtc True
```

### Open the client in a browser

```bash
firefox http://127.0.0.1:8080/
brave-browser-nightly --new-window http://127.0.0.1:8080/
```

Available settings in the UI:

- **Video resolution:** 320x240, 640x480, 960x540, 1280x720, 1920x1080
- **Video codec:** VP8, H264

### Stop the pipeline

1. Press **Ctrl+D** in the terminal running the client.
2. In the browser UI, click **Stop**, then **Exit API**.


### Replay recordings

```bash
#Setup config file
CONFIG_YAML=config_webrtc_ready_template.yaml

# Raw replay
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG replayer_raw False

# Inference replay
bash /workspace/volumes/ready/scripts/apis/webrtc_ready.bash ${CONFIG_YAML} LOCAL DEBUG replayer_inference False
```
