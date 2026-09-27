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
CONVERTER="${workspace_rootPath}/src/ready/apis/holoscan/utils/convert_video_to_gxf_entities.py"
[[ -f "$CONVERTER" ]] || fail "Converter script not found: $CONVERTER"

INPUT_VIDEO="${recorder_directory}/${recorder_basename}_640x400_timeframebound_cropresize_scale.mp4"
BASENAME_VIDEO="${recorder_directory}/${recorder_basename}${recorder_basenamePostProcessed}"


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
#pix_fmt rgb24
#pix_fmt rgba: Outputs 4-channel RGBA
#pix_fmt yuv420p
# Resolution is not standard — 640x400 is fine for RGB24, 
# but if your GXF converter expects standardized HD/4K or macro-block alignment, you may need to pad to 640x416 (divisible by 16) using:
# -vf "pad=640:416:0:8:black"
# --width 640 --height 416 --channels 3 --framerate 30
log "Converting video to GXF entities -> ${BASENAME_VIDEO}"
ffmpeg -y -loglevel error -i ${INPUT_VIDEO} \
    -fps_mode passthrough \
    -pix_fmt rgb24 -f rawvideo pipe:1 | \
python "$CONVERTER" \
    --width "$recorder_videoWidth" \
    --height "$recorder_videoHeight" \
    --channels "$recorder_videoChannels" \
    --framerate "$recorder_videoFrameRate" \
    --basename ${BASENAME_VIDEO}
# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "--------------------------------------"
log "Converted video: ${BASENAME_VIDEO}"
log "--------------------------------------"
 
