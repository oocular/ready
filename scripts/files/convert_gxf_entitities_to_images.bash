#!/bin/bash
set -Ee
## USAGE
# bash convert_gxf_entitities_to_images.bash


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../

source scripts/functions/parse_yaml.bash
eval $(parse_yaml configs/apis/config_webrtc_ready.yaml)

source .venv/bin/activate
cd src/ready/apis/holoscan/utils

if mkdir -p ${recorder_local_output_directory}; then
    log_info "Directory created or already exists: ${recorder_local_output_directory}"
else
    echo "Failed to create directory: ${recorder_local_output_directory}" >&2
    exit 1
fi


log_info "Creating output directory: ${recorder_local_output_directory}"
START_TIME=$(date +%s.%N)

python convert_gxf_entities_to_images.py \
--directory ${recorder_local_directory} \
--basename ${recorder_basename} \
--outputname ${recorder_local_imagefilname}${recorder_basename} \
--outputdir ${recorder_local_output_directory}

EXIT_CODE=$?
END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Conversion completed successfully"
    printf "Elapsed time: %.2f seconds\n" "$ELAPSED_TIME"
else
    log_error "Conversion failed with exit code: $EXIT_CODE"
    printf "Elapsed time: %.2f seconds\n" "$ELAPSED_TIME"
    exit $EXIT_CODE
fi

log_success "Script completed successfully"
