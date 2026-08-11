# Download holoscan utils
# Usage: bash holohub-utils-dependencies.bash
# LAUNCH PYTHON VIRTUAL ENV source .venv/bin/activate

download_if_missing() {
    local url="$1"
    local out="$2"
    if [ -f "$out" ]; then
        echo "Skipping $out (already exists)"
    else
        wget "$url" -O "$out"
    fi
}

download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/analyze.py analyze.py
download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/log_parser.py log_parser.py
download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/bar_plot_avg_datewise.py bar_plot_avg_datewise.py
download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/benchmarks/holoscan_flow_benchmarking/app_perf_graph.py app_perf_graph.py
download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/utilities/convert_video_to_gxf_entities.py convert_video_to_gxf_entities.py
download_if_missing https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/utilities/gxf_entity_codec.py gxf_entity_codec.py
