# webrtc
**TODO**: Create a config file to put all files and paths to easily user bashs scripts!

## Flow benchmarking
* Run [api_webrtc_ready](../../../docs/holoscan/apis_webrtc_ready.md) to generate logs
```
bash webrtc_ready.bash logweb_320x240Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_640x480Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_960x540Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_1280x720Mob1m.log PUBLIC OFF
bash webrtc_ready.bash logweb_1920x1080Mob1m.log PUBLIC OFF
```

* Analyse single logfiles
```
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_320x240Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_640x480Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_960x540Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_1280x720Mob1m.log
bash ../../../scripts/flow_benchmarking/A_analyse_logfile.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_1920x1080Mob1m.log
```

* B analyses multiple log files
```
bash ../../../scripts/flow_benchmarking/B_analyse_bar_plot_avg_datewise.bash $HOME/datasets/ready/webrtc/logsmobile-010324 logweb_320x240Mob1m.log  logweb_640x480Mob1m.log  logweb_960x540Mob1m.log logweb_1280x720Mob1m.log logweb_1920x1080Mob1m.log  
```

* Graph with Latency Numbers
```
#temrinal 1
bash ../../../scripts/flow_benchmarking/C_app_perf_graph.bash $HOME/datasets/ready/webrtc/logsmobile-010324

#terminal 2
bash ../../../scripts/flow_benchmarking/D_live_app_graph.bash $HOME/datasets/ready/webrtc/logsmobile-010324
```

* Results

Basic webrtc
![fig](plots.svg)

Full webrtc
![fig](plots_mobile.svg)
