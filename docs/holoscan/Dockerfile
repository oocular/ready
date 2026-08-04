ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG OPEN_CV_VERSION=4.10.0
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


RUN apt-get update && apt-get install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm \
    libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libjsoncpp-dev ffmpeg \
     && apt-get clean

RUN chmod +rwx /usr/bin/python3.10

# Install WebRTC Video Client (aiohttp aiortc) python modules
RUN pip3 install aiohttp aiortc opencv-python-headless #opencv-python for full version

