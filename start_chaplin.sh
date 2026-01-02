#!/bin/bash

# 1. Wait just enough for the Desktop (Reduced from 10s to 5s)
sleep 5

# 2. Move to project folder
cd /home/fawstech/vsr_v2/chaplin

# 3. FAST START: Use the virtual environment Python directly
# This skips the 'uv' dependency check which saves ~5-10 seconds
/home/fawstech/vsr_v2/chaplin/.venv/bin/python main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe > boot_log.txt 2>&1

# 4. Error handling (same as before)
if [ $? -ne 0 ]; then
    echo "------------------------------------------------"
    echo "CRITICAL ERROR: Chaplin crashed!"
    echo "------------------------------------------------"
    cat boot_log.txt
    echo "------------------------------------------------"
    echo "Press ENTER to close..."
    read input
fi

