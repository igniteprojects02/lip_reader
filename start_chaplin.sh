#!/bin/bash

# --- IMPROVEMENT 1: Say "Powering Up" IMMEDIATELY ---
# We don't wait for the screen or Python here. We just speak.
espeak -s 160 "Power on. Loading system." &

# --- IMPROVEMENT 2: Wait for Display (Background Loading) ---
# While the user hears the voice, we wait for the desktop
count=0
while ! xset q &>/dev/null; do
    sleep 0.5
    count=$((count+1))
    if [ $count -ge 60 ]; then break; fi # Wait max 30 seconds
done

cd /home/fawstech/vsr_v2/chaplin

# --- IMPROVEMENT 3: Rotate Logs ---
# Don't delete the old log immediately; keep a backup just in case
mv boot_log.txt boot_log_prev.txt 2>/dev/null

# --- LAUNCH APP ---
# Using the direct Python path for speed
/home/fawstech/vsr_v2/chaplin/.venv/bin/python main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe > boot_log.txt 2>&1

# Error Handling
if [ $? -ne 0 ]; then
    echo "------------------------------------------------"
    echo "CRITICAL ERROR: Chaplin crashed!"
    echo "------------------------------------------------"
    cat boot_log.txt
    echo "------------------------------------------------"
    echo "Press ENTER to close..."
    read input
fi
