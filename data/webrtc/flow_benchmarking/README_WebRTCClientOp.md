# webrtc with mobile

## Simple workflow

### Flow benchmarking
* Run [api_webrtc_ready](../../../docs/holoscan/apis_webrtc_ready.md) to generate logs
```
bash webrtc_ready.bash logweb_320x240Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_640x480Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_960x540Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_1280x720Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_1920x1080Mob1m.log PUBLIC OFF
```

* Create path and move for files
```
mkdir -p logsmobile-200324
mv logweb_* logsmobile-200324
```


* Analyse single logfiles
```
cd $HOME/repositories/oocular/ready/data/webrtc/flow_benchmarking
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_320x240Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_640x480Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_960x540Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_1280x720Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_1920x1080Mob1m.log
```

* B analyses multiple log files
```
bash ../../../scripts/flow_benchmarking/B_analyse_bar_plot_avg_datewise.bash $HOME/datasets/ready/webrtc/logsmobile-200324 logweb_320x240Mob1m.log  logweb_640x480Mob1m.log  logweb_960x540Mob1m.log logweb_1280x720Mob1m.log logweb_1920x1080Mob1m.log
```

* Graph with Latency Numbers
```
#temrinal 1
bash ../../../scripts/flow_benchmarking/C_app_perf_graph.bash $HOME/datasets/ready/webrtc/logsmobile-200324

#terminal 2
bash ../../../scripts/flow_benchmarking/D_live_app_graph.bash $HOME/datasets/ready/webrtc/logsmobile-200324
```

* Results

Basic flow webrtc
![fig](plots_mobile_basicflow.svg)


## WebClientOp>DropFramesOp>VisualizerSinkOp workflow

### Flow benchmarking

* Change values of `branch_hz = 1` in webrtc_client.py to 1, 2, 3, 4, 5.
* Create path and move for files where $X is the value: 1, 2, 3, 4, 5.
```
X=1
mkdir -p logsmobile-210324-branch_hz_${X}
mv logweb_* logsmobile-210324-branch_hz_${X}
```

* Run [api_webrtc_ready](../../../docs/holoscan/apis_webrtc_ready.md) to generate logs
```
bash webrtc_ready.bash logweb_0320x0240Mob1min.log PUBLIC OFF
bash webrtc_ready.bash logweb_0640x0480Mob1min.log PUBLIC OFF
bash webrtc_ready.bash logweb_0960x0540Mob1min.log PUBLIC OFF
bash webrtc_ready.bash logweb_1280x0720Mob1min.log PUBLIC OFF
bash webrtc_ready.bash logweb_1920x1080Mob1min.log PUBLIC OFF
```


* Analyse single logfiles
```
cd $HOME/repositories/oocular/ready/data/webrtc/flow_benchmarking
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_0320x0240Mob1min.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_0640x0480Mob1min.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_0960x0540Mob1min.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_1280x0720Mob1min.log 
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_1920x1080Mob1min.log
```

* B analyses multiple log files
```
bash ../../../scripts/flow_benchmarking/B_analyse_bar_plot_avg_datewise.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5 logweb_0320x0240Mob1min.log logweb_0640x0480Mob1min.log logweb_0960x0540Mob1min.log logweb_1280x0720Mob1min.log logweb_1280x0720Mob1min.log
```

* Graph with Latency Numbers
```
#temrinal 1
bash ../../../scripts/flow_benchmarking/C_app_perf_graph.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5

#terminal 2
bash ../../../scripts/flow_benchmarking/D_live_app_graph.bash $HOME/datasets/ready/webrtc/logsmobile-210324-branch_hz_5
```

* Results

Average end-to-end latency 1/2=0.5 (500ms)
![fig](plots_mobile_branch_hz_2.svg)

Average end-to-end latency 1/5=0.2 (200ms)
![fig](plots_mobile_branch_hz_5.svg)


