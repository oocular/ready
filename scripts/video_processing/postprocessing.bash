#!/bin/bash
#
# Video scaling and time-frame trimming, cropping and resizing utilites
#
# Usage:
#   bash postprocessing.bash <config_file_name>
#
# Example:
#   bash postprocessing.bash config_webrtc_ready_template.yaml
#
set -Ee

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '%s\n' "$*"; }

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Defaults (can be overridden via environment variables)
# ---------------------------------------------------------------------------
: "${WIDTH:=640}"
: "${HEIGHT:=400}"
: "${START_FRAME_TIME:=1.5}"
: "${END_FRAME_TIME:=5.1}"
: "${FFMPEG_BIN:=ffmpeg}"


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

# Map values from config file
WIDTH="$recorder_videoWidth"
HEIGHT="$recorder_videoHeight"
START_FRAME_TIME="$recorder_videoStartFrameTime"
END_FRAME_TIME="$recorder_videoEndFrameTime"

# ---------------------------------------------------------------------------
# Paths and output
# ---------------------------------------------------------------------------
PATH_VIDEO="${recorder_directory}/${recorder_basename}.mp4"
PATH_OUTPUT_VIDEO="${recorder_directory}/${recorder_basename}_${WIDTH}x${HEIGHT}.mp4"
PATH_OUTPUT_TFB_VIDEO="${recorder_directory}/${recorder_basename}_${WIDTH}x${HEIGHT}_timeframebound.mp4"
PATH_OUTPUT_CR_VIDEO="${recorder_directory}/${recorder_basename}_${WIDTH}x${HEIGHT}_timeframebound_cropresize.mp4"
PATH_OUTPUT_S_VIDEO="${recorder_directory}/${recorder_basename}_${WIDTH}x${HEIGHT}_timeframebound_cropresize_scale.mp4"


# ---------------------------------------------------------------------------
# Scale video to WIDTHxHEIGHT
# ---------------------------------------------------------------------------
log "Scaling video to ${WIDTH}x${HEIGHT} ..."
"$FFMPEG_BIN" -y -i "$PATH_VIDEO" \
  -vf "scale=${WIDTH}:${HEIGHT}" \
  "$PATH_OUTPUT_VIDEO"

log "--------------------------------------"
log "Converted video: ${PATH_OUTPUT_VIDEO}"
log "--------------------------------------"


# ---------------------------------------------------------------------------
# Compute duration safely
# ---------------------------------------------------------------------------
DURATION_FRAME="$(awk -v s="$START_FRAME_TIME" -v e="$END_FRAME_TIME" \
  'BEGIN { d = e - s; if (d <= 0) { print "ERR"; exit 1 } printf "%.1f", d }')" \
  || die "END_FRAME_TIME (${END_FRAME_TIME}) must be greater than START_FRAME_TIME (${START_FRAME_TIME})"


# ---------------------------------------------------------------------------
# Cut timeframes of the video
# ---------------------------------------------------------------------------
log "Trimming video from ${START_FRAME_TIME}s for ${DURATION_FRAME}s ..."
"$FFMPEG_BIN" -y \
  -ss "$START_FRAME_TIME" \
  -i "$PATH_OUTPUT_VIDEO" \
  -vcodec libx264 \
  -acodec copy \
  -t "$DURATION_FRAME" \
  "$PATH_OUTPUT_TFB_VIDEO"

log "--------------------------------------"
log "Converted video: ${PATH_OUTPUT_TFB_VIDEO}"
log "--------------------------------------"


# ---------------------------------------------------------------------------
# Cropped video
# ---------------------------------------------------------------------------
# ffmpeg -y -i "$PATH_OUTPUT_TFB_VIDEO" \
#   -vf "crop=${recorder_cropVideoWidth}:${recorder_cropVideoHeight}:${recorder_crop_xpos}:${recorder_crop_ypos}" \
#   -c:a copy "$PATH_OUTPUT_CR_VIDEO"
log "Cropping ${recorder_cropVideoWidth}x${recorder_cropVideoHeight} at (${recorder_crop_xpos},${recorder_crop_ypos}) ..."
"$FFMPEG_BIN" -y \
  -i "$PATH_OUTPUT_TFB_VIDEO" \
  -vf "crop=${recorder_cropVideoWidth}:${recorder_cropVideoHeight}:${recorder_crop_xpos}:${recorder_crop_ypos}" \
  -c:a copy \
  "$PATH_OUTPUT_CR_VIDEO"
log "--------------------------------------"
echo "Crop: ${recorder_cropVideoWidth}x${recorder_cropVideoHeight} at (${recorder_crop_xpos},${recorder_crop_ypos})"
log "Cropped video:  ${PATH_OUTPUT_CR_VIDEO}"
log "--------------------------------------"


# ---------------------------------------------------------------------------
# Scale cropped video to final size
# ---------------------------------------------------------------------------
log "Scaling to ${recorder_videoWidth}x${recorder_videoHeight} ..."
"$FFMPEG_BIN" -y \
  -i "$PATH_OUTPUT_CR_VIDEO" \
  -vf scale=${recorder_videoWidth}:${recorder_videoHeight} \
  "$PATH_OUTPUT_S_VIDEO"
log "--------------------------------------"
log "Scaled video:   ${PATH_OUTPUT_S_VIDEO}"
log "--------------------------------------"



# ---------------------------------------------------------------------------
# Remove files
# ---------------------------------------------------------------------------
rm $PATH_OUTPUT_VIDEO #${recorder_basename}_${WIDTH}x${HEIGHT}.mp4"
rm $PATH_OUTPUT_TFB_VIDEO #${recorder_basename}_${WIDTH}x${HEIGHT}_timeframebound.mp4"
rm $PATH_OUTPUT_CR_VIDEO #${recorder_basename}_${WIDTH}x${HEIGHT}_timeframebound_cropresize.mp4"
