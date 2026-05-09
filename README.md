# MTIL-LeRobot

A LeRobot implementation of **Mamba Temporal Imitation Learning (MTIL)**,
based on [Zhou et al., "Encoding Full History with Mamba for Temporal Imitation Learning"](https://arxiv.org/abs/2505.12410)
(IEEE RA-L, 2025).

MTIL uses a Mamba state-space backbone to encode the full episode history, enabling policies to disambiguate visually identical states that differ only in temporal context.

## Installation

Clone with submodules to install the policy package with the bundled LeRobot fork:

```bash
git clone --recurse-submodules https://github.com/hovak101/MTIL-Lerobot.git
cd MTIL-Lerobot
pip install -e ./lerobot
pip install -e ./lerobot_policy_mtil
```

For asynchronous inference (running the policy on a separate host from the robot), also install the async extras:

```bash
pip install -e './lerobot[async]'
```

## Usage

This repo plugs into the standard LeRobot CLI. The typical workflow is:

1. **Discover hardware** — find serial ports for your arms and indices for your cameras (`lerobot-find-port`, `lerobot-find-cameras`).
2. **Teleoperate** — verify your leader/follower setup with `lerobot-teleoperate`.
3. **Record demonstrations** — collect a dataset with `lerobot-record`, pushing to the Hugging Face Hub.
4. **Train** — train an MTIL policy on your dataset using LeRobot's training entry point.
5. **Roll out** — deploy the trained policy with `lerobot-rollout`. The `sentry` strategy continuously runs the policy and records episodes; `base` runs the policy without recording for quick evaluation.

Please refer to the [official lerobot documentation](https://huggingface.co/docs/lerobot/index) for more details

### Remote inference

To run the policy on a more powerful machine over LAN, launch the policy server on the inference host and the rollout client on the robot host. MTIL requires CUDA enabled gpu for inference. See `scripts/serve.sh` and `scripts/mtil_rollout_client.sh` for templates.

## Results
The resources folder contains a report and you can find our presentation on [Google Slides](https://docs.google.com/presentation/d/1D5RGEx_oHiUDR4ooh0IAy3R7szCcRgLGBlwVf5lmwUc/edit?pli=1&slide=id.g3dd8f6a9490_0_396#slide=id.g3dd8f6a9490_0_396). Furthermore, you can find our recorded datatsets and model on [Hugging Face](https://huggingface.co/hovak101). Please note MTIL does not include an eval dataset because we ran inference through LAN, which does not support recording.

## Citation

If you use this code, please cite the original MTIL paper:

```bibtex
@article{Zhou2025MTIL,
  author={Zhou, Yulin and Lin, Yuankai and Peng, Fanzhe and Chen, Jiahui and Huang, Kaiji and Yang, Hua and Yin, Zhouping},
  journal={IEEE Robotics and Automation Letters},
  title={MTIL: Encoding Full History with Mamba for Temporal Imitation Learning},
  year={2025},
  volume={10},
  number={11},
  pages={11761-11767},
  doi={10.1109/LRA.2025.3615520}
}
```

Original implementation: [yulinzhouZYL/MTIL](https://github.com/yulinzhouZYL/MTIL)
