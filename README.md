# MTIL-LeRobot

A LeRobot implementation of **Mamba Temporal Imitation Learning (MTIL)**, 
based on [Zhou et al., "Encoding Full History with Mamba for Temporal Imitation Learning"](https://arxiv.org/abs/2505.12410) 
(IEEE RA-L, 2025).

## Setup

```bash
git clone --recurse-submodules https://github.com/hovak101/cs152_final_project.git
```

### If Processing Inference Through LAN PC:
```bash
pip install -e 'lerobot[async]'
```

#### On server:
##### find ip:
```bash
hostname -I | awk '{print $1}'
```
##### launch:
```bash
bash scripts/serve_mtil.sh
```

#### On client:
```bash
bash scripts/rollout_client.sh <ip_address>
```

## Data Collection + Inference Commands

### find arm port:
```bash
lerobot-find-port
```

### find camera ports: 
```bash
lerobot-find-cameras opencv
```

### Set HF_USER
```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==2 {print $2}')     
echo $HF_USER
```            

### test teleop
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044252-if00 \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044921-if00 \
    --teleop.id=my_leader
```
### record episodes
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044252-if00 \
    --robot.id=my_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_F4885D8F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}, front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044921-if00 \
    --teleop.id=my_leader \
    --display_data=true \
    --dataset.root=data \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=50 \
    --dataset.single_task="Place red ball in bergundy bowl." \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.vcodec=h264
```

### delete cache folder
```bash
rm -rf /home/alex/.cache/huggingface/lerobot/${HF_USER}/record-test
```

### run inference:
`lerobot-record` is data-collection only in this version — use `lerobot-rollout` for policy deployment. Sentry strategy continuously runs the policy and records episodes (rotated by video file size, not time).
```bash
lerobot-rollout \
    --strategy.type=sentry \
    --policy.path=hovak101/my_mtil_policy \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044252-if00 \
    --robot.id=my_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_F4885D8F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}, front: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG, warmup_s: 5}}" \
    --dataset.repo_id=hovak101/rollout_mtil \
    --dataset.single_task="Place red ball in bergundy bowl." \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.vcodec=h264 \
    --duration=300 \
    --display_data=true
```
For a quick eval without recording, use `--strategy.type=base` (drop all `--dataset.*` flags) and set `--duration` / `--task`.