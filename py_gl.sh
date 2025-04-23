#!/bin/bash

# Start Xvfb if not already running
XVFB_PID=$(pgrep Xvfb)
if [ -z "$XVFB_PID" ]; then
    echo "Starting Xvfb..."
    Xvfb :99 -screen 0 1400x900x24 -nolisten tcp &
    sleep 1
fi

# Export all necessary env vars
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Qt headless fix
export QT_QPA_PLATFORM=offscreen

# Activate your Conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rarl_new

# Run the script
python "$@"
