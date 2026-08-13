#!/bin/bash
# Wait for vision to finish, then run memorization + language + generate.
cd /home/god/Desktop/whitebox
while pgrep -f "exp_vision.py --epochs 30" > /dev/null; do
    sleep 5
done
echo "=== vision done at $(date +%H:%M:%S) ===" >> watchdog.log
bash run_after_vision.sh 2>> watchdog.log
echo "=== watchdog finished at $(date +%H:%M:%S) ===" >> watchdog.log
