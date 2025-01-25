# webrtc
**TODO**: Create a config file to put all files and paths to easily user bashs scripts!

## Flow benchmarking
* Run app to generate logs
```
bash webrtc.bash LOCAL logger_local_W320H240vp8.log
bash webrtc.bash LOCAL logger_local_W320H240h264.log
bash webrtc.bash LOCAL logger_local_W1920H1080vp8.log
bash webrtc.bash LOCAL logger_local_W1920H1080h264.log
#640x480
#960x540
#1280x720
#1920x1080
```

* Analyse single logfiles
```
bash ../../scripts/flow_benchmarking/A_analyse_logfile.bash logger_local_W320H240vp8.log $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
bash ../../scripts/flow_benchmarking/A_analyse_logfile.bash logger_local_W320H240h264.log $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
bash ../../scripts/flow_benchmarking/A_analyse_logfile.bash logger_local_W1920H1080vp8.log $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
bash ../../scripts/flow_benchmarking/A_analyse_logfile.bash logger_local_W1920H1080h264.log $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
```
* B analyses multiple log files
```
bash ../../scripts/flow_benchmarking/B_analyse_bar_plot_avg_datewise.bash $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs logger_local_W320H240vp8.log logger_local_W320H240h264.log logger_local_W1920H1080vp8.log logger_local_W1920H1080h264.log
```

* Graph with Latency Numbers
```
#temrinal 1
bash ../../scripts/flow_benchmarking/C_app_perf_graph.bash $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
#terminal 2
bash ../../scripts/flow_benchmarking/D_live_app_graph.bash $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
```

* Results
![fig](plots.svg)
