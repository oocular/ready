ARG BASE_IMAGE=holoscan-sdk-build-cu12-x86_64:latest
FROM ${BASE_IMAGE} AS base

#############################################################
# Install common dependencies via HoloHub CLI
#############################################################
ENV DEBIAN_FRONTEND=noninteractive

FROM base AS holohub-setup
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

#############################################################
# Install Runtime Dependencies
#############################################################
FROM holohub-setup AS run

RUN apt-get update && apt-get install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm \
    libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libjsoncpp-dev ffmpeg \
     && apt-get clean


# Install WebRTC Video Client (aiohttp aiortc) python modules
RUN python3 -m pip install aiohttp aiortc opencv-python-headless
