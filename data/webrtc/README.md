# webrtc

## Flow benchmarking
* Analyse single logfiles
```
bash ../../scripts/flow_benchmarking/A_analyse_logfile.bash logger $HOME/repositories/ready/src/ready/apis/holoscan/webrtc
```
* B analyses multiple log files
```
bash ../../scripts/flow_benchmarking/B_analyse_bar_plot_avg_datewise.bash logger_local logger_public $HOME/repositories/ready/src/ready/apis/holoscan/webrtc
```
* C_app_perf_graph.bash
```
bash ../../scripts/flow_benchmarking/C_app_perf_graph.bash $HOME/repositories/ready/src/ready/apis/holoscan/webrtc/logs
bash ../../scripts/flow_benchmarking/D_live_app_graph.bash
```
