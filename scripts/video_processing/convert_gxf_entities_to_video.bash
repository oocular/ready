#!/bin/bash
#
# Convert GXF entities to an MP4 video.
#
# Usage:
#   bash convert_gxf_entities_to_video.bash <config_yaml>
#
# Example:
#   bash convert_gxf_entities_to_video.bash config_webrtc_ready_template.yaml
#
set -Ee

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CONFIG_FILE_NAME="$1"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH_FILE="${REPO_ROOT}/configs/apis/${CONFIG_FILE_NAME}"
PARSE_YAML="${REPO_ROOT}/scripts/functions/parse_yaml.bash"


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "$PARSE_YAML"
eval "$(parse_yaml "$CONFIG_PATH_FILE")"

cd "$workspace_rootPath"

# ---------------------------------------------------------------------------
# Paths and output
# ---------------------------------------------------------------------------
CONVERTER="${workspace_rootPath}/src/ready/apis/holoscan/utils/convert_gxf_entities_to_video.py"
[[ -f "$CONVERTER" ]] || fail "Converter script not found: $CONVERTER"

OUTPUT_VIDEO="${recorder_directory}/${recorder_basename}.mp4"
mkdir -p "$recorder_directory"

# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
log "Converting GXF entities -> ${OUTPUT_VIDEO}"
python "$CONVERTER" \
    --basename  "$recorder_basename" \
    --directory "$recorder_directory" \
| ffmpeg \
    -f rawvideo \
    -pix_fmt rgb24 \
    -s "$recorder_videoSensorWidth"x"$recorder_videoSensorHeight" \
    -r "$recorder_videoSensorFrameRate" -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -r "$recorder_videoSensorFrameRate" \
    -y "$OUTPUT_VIDEO"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "--------------------------------------"
log "Converted video: ${OUTPUT_VIDEO}"
log "--------------------------------------"