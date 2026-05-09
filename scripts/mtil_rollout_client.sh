#!/usr/bin/env bash
# Robot client — run this on the laptop connected to the robot and cameras.
#
# Usage:
#   bash scripts/rollout_client.sh <WSL_IP>
#
# Get the WSL IP by running on the GPU machine:
#   hostname -I | awk '{print $1}'

set -euo pipefail

SERVER_IP="${1:?Error: provide the GPU machine IP as the first argument.
Usage: bash scripts/rollout_client.sh <WSL_IP>}"

python -m lerobot.async_inference.robot_client \
    --policy_type=mtil \
    --pretrained_name_or_path=/home/alex/projects/MTIL-Lerobot/outputs/train/mtil_record_test_4/checkpoints/007500/pretrained_model \
    --server_address="${SERVER_IP}:8080" \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044252-if00 \
    --robot.id=my_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_F4885D8F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}, front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}}" \
    --task="Place red ball in bergundy bowl." \
    --actions_per_chunk=50 \
    --policy_device=cuda \
    --client_device=cpu \
    --fps=30 \
    --aggregate_fn_name=weighted_average \
    --chunk_size_threshold=0.5
