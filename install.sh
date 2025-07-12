#!/bin/bash
set -e

# use runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 docker images in runpod

# apt packages
apt-get update && apt-get -y install libopenmpi-dev python3-venv nano htop ffmpeg
cd /workspace
git clone https://github.com/taresh18/parakeet-FastAPI.git && cd parakeet-FastAPI
python3 -m venv parakeet && source parakeet/bin/activate
pip install numpy typing_extensions # required for sox build
pip install -r requirements.txt
chmod +x start.sh && bash -x start.sh
