# Download holoscan utils
# Usage: bash holohub-utils-dependencies.sh

# (1) LAUNCH PYTHON VIRTUAL ENV
# pip install nvidia-pyindex
python -m ensurepip --default-pip
python -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

#TODO check paths for scripts
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/analyze.py -O analyze.py
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/log_parser.py -O log_parser.py
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/bar_plot_avg_datewise.py -O bar_plot_avg_datewise.py
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/app_perf_graph.py -O app_perf_graph.py
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/utilities/convert_video_to_gxf_entities.py -O convert_video_to_gxf_entities.py
wget https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/utilities/gxf_entity_codec.py -O gxf_entity_codec.py
