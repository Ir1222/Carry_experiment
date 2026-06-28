<h1 align="center"> PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System </h1>

<p align="center">
  <a href='https://why618188.github.io/' target='_blank'>Huayi Wang*</a>,
  <a href='https://zwt006.github.io/' target='_blank'>Wentao Zhang*</a>,
  <a href='https://ingrid789.github.io/IngridYu/' target='_blank'>Runyi Yu*</a>,
  <a href="https://taohuang13.github.io/">Tao Huang</a>,
  <a href="https://renjunli99.github.io/">Junli Ren</a>,
  <a href="https://trap-1.github.io/">Feiyu Jia</a>,
  <a href="https://openreview.net/profile?id=%7EZiRui_Wang4">Zirui Wang</a>,
  <br>
  <a href="https://why618188.github.io/physhsi/">Xiaojie Niu</a>,
  <a href="https://xiao-chen.tech/">Xiao Chen</a>,
  <a href="https://jiahe-chen.cn/">Jiahe Chen</a>,
  <a href="https://cqf.io/">Qifeng Chen<sup>&dagger;</sup></a>,
  <a href="https://wangjingbo1219.github.io/">Jingbo Wang<sup>&dagger;</sup></a>,
  <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang<sup>&dagger;</sup></a>
  <br>
  *Equal Contributions&nbsp;&nbsp;&nbsp;&nbsp;<sup>&dagger;</sup>Corresponding Authors
  <br>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.11072"><img src="https://img.shields.io/badge/arXiv-2510.11072-brown" alt="arXiv"></a>
  <a href="https://youtu.be/dTj6FjoQ5u0"><img src="https://img.shields.io/badge/Youtube-🎬-yellow" alt="YouTube"></a>
  <a href="https://why618188.github.io/physhsi/"><img src="https://img.shields.io/badge/Website-%F0%9F%9A%80-green" alt="Website"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0"></a>
</p>

<p align="center">
  <img src="teaser.png" alt="Project teaser" width="100%">
</p>

## 🛠️ Installation

1. Create a conda environment and install PyTorch:

```bash
conda create -n physhsi python=3.8
conda activate physhsi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. Install Isaac Gym:
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Navigate to its Python folder and install.
  ```bash
  cd isaacgym/python && pip install -e .
  ```

3. Clone this repository.
```bash
git clone https://github.com/InternRobotics/PhysHSI.git
cd PhysHSI
```

4. Install PhysHSI.
```bash
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
```

5. Install additional dependencies.
```bash
cd .. && pip install -r requirements.txt
```

## 🕹️ Run PhysHSI

PhysHSI supports six tasks for the Unitree G1 humanoid robot: **CarryBox, SitDown, LieDown, StandUp, StyleLoco-Dinosaur,** and **StyleLoco-Highknee**.

### Motion Visualization

Reference motion data for each task can be found in the [motion data folder](legged_gym/resources/dataset/). To visualize reference motion data, run:

```bash
cd legged_gym
python legged_gym/scripts/play.py --task [task_name] --play_dataset
```

Here, `[task_name]` can be one of `[carrybox, liedown, sitdown, standup, styleloco_dinosaur, styleloco_highknee]`.

### Play with Pre-trained Checkpoints

Pre-trained checkpoints for each task are available in the [checkpoint folder](legged_gym/resources/ckpt/). To play a task using a checkpoint, run:

```bash
python legged_gym/scripts/play.py --task [task_name] --resume_path resources/ckpt/[task_name].pt
```

For example, to play the CarryBox task:
```bash
python legged_gym/scripts/play.py --task carrybox --resume_path resources/ckpt/carrybox.pt
```

> ⚠️ Note:
> 
> During the first 1–2 episodes of `play.py`, you may observe slight interpenetration between the robot, the object, or the platform.
> 
> This issue only occurs in the initial episodes and does not affect training or subsequent performance.

## 🤖 Train PhysHSI

### CarryBox

CarryBox is a challenging long-horizon task. The current `carrybox` and
`carrybox_resume` configurations use a 738-D actor history and a compact 143-D
critic: the original PhysHSI 126-D current-frame critic plus the existing 17-D
privileged interaction proxy. No additional command-line flag is required.

Run the following Ubuntu commands from the PhysHSI repository root.

1. **Initial Phase A training (20,000 iterations):**

    ```bash
    conda activate physhsi
    cd /absolute/path/to/PhysHSI
    python legged_gym/legged_gym/scripts/train.py \
        --task carrybox \
        --headless \
        --rl_device cuda:0 \
        --run_name phase_a_stage1 \
        --max_iterations 20000
    ```

    TensorBoard logs and checkpoints are written under
    `legged_gym/logs/amp_carrybox/<timestamp>_phase_a_stage1/`. The final
    checkpoint is normally named `model_19999.pt` because checkpoint filenames
    use a zero-based learning-iteration index.

2. **Refined Phase A training (30,000 additional iterations):**

    Replace `<timestamp>` below with the timestamp in the stage-1 run directory.
    The resume checkpoint must have been trained with the current 143-D critic.
    Old 758-D actor-history-conditioned critic checkpoints are intentionally
    shape-incompatible and must only be used from the
    `backup/critic-history-758d` branch or the `critic-history-758d-v1` tag.

    ```bash
    STAGE1_CKPT="$PWD/legged_gym/logs/amp_carrybox/<timestamp>_phase_a_stage1/model_19999.pt"
    test -f "$STAGE1_CKPT"

    python legged_gym/legged_gym/scripts/train.py \
        --task carrybox_resume \
        --resume \
        --resume_path "$STAGE1_CKPT" \
        --headless \
        --rl_device cuda:0 \
        --run_name phase_a_stage2 \
        --max_iterations 30000
    ```

    Since stage 2 resumes at iteration 20,000, its final checkpoint is normally
    `legged_gym/logs/amp_carrybox/<timestamp>_phase_a_stage2/model_49999.pt`.

3. **Visual validation with `play.py`:**

    Replace `<timestamp>` with the stage-2 run timestamp. Do not pass
    `--headless`; `play.py` opens the Isaac Gym viewer. The critic and privileged
    observations are used during training only; playback executes the unchanged
    actor from the Phase A checkpoint.

    ```bash
    FINAL_CKPT="$PWD/legged_gym/logs/amp_carrybox/<timestamp>_phase_a_stage2/model_49999.pt"
    test -f "$FINAL_CKPT"

    python legged_gym/legged_gym/scripts/play.py \
        --task carrybox \
        --resume_path "$FINAL_CKPT" \
        --rl_device cuda:0 \
        --num_envs 1
    ```

### Other Tasks

For the remaining five tasks, you can directly train them using:
```bash
python legged_gym/scripts/train.py --task [task_name] --headless
```
Here, `[task_name]` can be one of `[liedown, sitdown, standup, styleloco_dinosaur, styleloco_highknee]`.

To play the final trained checkpoint for any task:
```bash
python legged_gym/scripts/play.py --task [task_name] --resume_path [ckpt_path]
```

> By default, PhysHSI uses **TensorBoard** for logging training metrics.  
> 
> If you prefer to use **Weights & Biases (wandb)**, please enable it in the corresponding `[task_name]_config.py` file and set the appropriate `wandb_entity` for your account.

For Phase A CarryBox training, TensorBoard also records the added critic-tail
channels under `Privileged/channels`, vector magnitudes under
`Privileged/norms`, and hand-contact diagnostics under `Privileged/contact`.
The contact flag mismatch rates should remain exactly zero, and
`Privileged/rollout_logged_steps` should equal the configured rollout length
(`100` by default).

From the PhysHSI repository root, start TensorBoard with:

```bash
tensorboard --logdir legged_gym/logs/amp_carrybox --port 6006 --host 127.0.0.1
```

Then open `http://127.0.0.1:6006`. For a remote Ubuntu server, forward the port
from the local computer before opening the same URL:

```bash
ssh -L 6006:127.0.0.1:6006 USER@SERVER_IP
```

## 👏 Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [Legged_gym](https://github.com/leggedrobotics/rsl_rl) and [HIMLoco](https://github.com/OpenRobotLab/HIMLoco): The foundation environments for training and running codes.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): Reinforcement learning algorithm implementation.
- [AMP for Hardware](https://github.com/escontra/AMP_for_hardware) and [TokenHSI](https://github.com/liangpan99/TokenHSI): References for AMP and RSI implementations.
- [AMASS](https://amass.is.tue.mpg.de/), [SAMP](https://samp.is.tue.mpg.de/) and [100STYLE](https://www.ianxmason.com/100style/): Reference dataset construction.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{wang2025physhsi,
  title   = {PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System},
  author  = {Wang, Huayi and Zhang, Wentao and Yu, Runyi and Huang, Tao and Ren, Junli and Jia, Feiyu and Wang, Zirui and Niu, Xiaojie and Chen, Xiao and Chen, Jiahe and Chen, Qifeng and Wang, Jingbo and Pang, Jiangmiao},
  journal = {arXiv preprint arXiv:2510.11072},
  year    = {2025},
}
```

## 📄 License

The PhysHSI code is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Commercial use is not allowed without explicit authorization.
