#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing system dependencies ---"
# Update package list and install ffmpeg
sudo apt-get update
sudo apt-get install -y ffmpeg

echo "--- Installing Python dependencies ---"
# Install Python dependencies from requirements.txt
pip install -r requirements.txt

echo "--- Build complete ---"