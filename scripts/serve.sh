#!/usr/bin/env bash
# Policy server — run this on the WSL GPU machine.
#
# Before starting: make sure the mtil package is installed:
#   pip install -e /path/to/cs152_final_project/lerobot_policy_mtil/
#
# Get this machine's IP to give to the laptop client:
#   hostname -I | awk '{print $1}'

set -euo pipefail

python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
