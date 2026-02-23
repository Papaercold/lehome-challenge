# Remote Training Deployment Guide

GH200 (ARM64 aarch64) + X-VLA full fine-tuning on LeHome Challenge dataset.

---

## Prerequisites

- Remote server: NVIDIA GH200 / H100 (ARM64 aarch64)
- Local machine: any (for code sync and evaluation)

---

## Step 1: SSH into Remote Server

```bash
ssh root@<SERVER_IP>
# or with key file:
ssh -i ~/.ssh/your_key.pem ubuntu@<SERVER_IP>
```

---

## Step 2: Install Miniforge (ARM64)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
echo 'source ~/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc
```

---

## Step 3: Create Python Environment

```bash
conda create -n xvla_train python=3.11 -y
conda activate xvla_train
```

---

## Step 4: Install PyTorch (ARM64 — must use pip, NOT conda)

```bash
# CUDA 12.6 wheel — works with CUDA 12.6/12.8 drivers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# Expected: CUDA: True 12.6 NVIDIA GH200 480GB
```

---

## Step 5: Install lerobot from Source (editable install)

Installing from source allows direct code modification without reinstalling.

```bash
# Clone lerobot source
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
git checkout v0.4.3        # pin to tested version
pip install -e ".[xvla]"   # editable install
cd ~

# Additional dependencies
pip install "transformers>=4.47.0,<5.0.0"
pip install draccus huggingface_hub
```

> `transformers` version must be pinned — required for Florence2LanguageConfig and huggingface-hub 1.x compatibility.

Verify:
```bash
python -c "import lerobot; print('lerobot OK')"
python -c "from transformers import AutoModel; print('transformers OK')"
```

---

## Step 6: Apply Gripper Fix

The default `so101_bimanual` action mode zeros out gripper channels during training,
so the gripper never learns to close. This patch enables gripper learning.

```bash
# Find the file location
ACTION_HUB=$(python -c "import lerobot.policies.xvla.action_hub as m; import inspect; print(inspect.getfile(m))")
echo $ACTION_HUB   # should be inside ~/lerobot/

# Apply patch
python -c "
path = '$ACTION_HUB'
with open(path) as f:
    lines = f.readlines()

out = []
skip_next = False
for i, line in enumerate(lines):
    # Remove gripper zeroing in preprocess
    if 'gripper_idx] = 0.0' in line:
        continue
    # Remove 'if action_m is not None:' guard before the zeroing
    if 'if action_m is not None:' in line and i+1 < len(lines) and 'gripper_idx] = 0.0' in lines[i+1]:
        continue
    # Remove sigmoid on gripper in postprocess
    if 'gripper_idx] = torch.sigmoid' in line:
        continue
    # Remove the 'if action.size(-1) > max' guard before sigmoid
    if 'if action.size(-1) > max(self.gripper_idx)' in line and i+1 < len(lines) and 'torch.sigmoid' in lines[i+1]:
        continue
    out.append(line)

with open(path, 'w') as f:
    f.writelines(out)
print('Gripper fix applied:', path)
"
```

Verify the fix was applied:
```bash
grep -n "gripper_idx] = 0.0\|torch.sigmoid" $ACTION_HUB
# Expected: no output (lines have been removed)
```

---

## Step 7: Clone Repository

```bash
git clone <YOUR_REPO_URL> ~/lehome-challenge
cd ~/lehome-challenge
```

---

## Step 8: Download Dataset

```bash
cd ~/lehome-challenge

# Login if dataset is private
huggingface-cli login

# Download merged dataset (no depth, all 4 garment types)
huggingface-cli download lehome/dataset_challenge_merged \
  --repo-type dataset \
  --local-dir Datasets/example
```

Verify:
```bash
ls Datasets/example/
# Expected: four_types_merged  pant_long_merged  pant_short_merged  top_long_merged  top_short_merged
```

> **Assets are NOT needed for training** — only for IsaacSim evaluation (run locally).

---

## Step 9: Start Training

```bash
cd ~/lehome-challenge
conda activate xvla_train

# Create tmux session to keep training alive after SSH disconnect
tmux new -s xvla

# Run training (first run auto-downloads xvla-base ~3.5GB from HuggingFace)
lerobot-train --config_path=configs/train_xvla.yaml
```

Detach from tmux (training keeps running): `Ctrl+B` then `D`

Re-attach to see logs: `tmux attach -t xvla`

---

## Step 10: Monitor Training

In a separate SSH window:
```bash
# GPU utilization
watch -n 5 nvidia-smi

# Training logs
tail -f ~/lehome-challenge/outputs/train/xvla_finetune_top_long_h100_v2/train.log
```

Expected loss curve: starts ~0.7, drops to ~0.05–0.03 over 100K steps.

---

## Step 11: Upload Checkpoint to HuggingFace (run on remote server)

```bash
cd ~/lehome-challenge

huggingface-cli upload <YOUR_HF_USERNAME>/<MODEL_REPO_NAME> \
  outputs/train/xvla_finetune_top_long_h100_v2/checkpoints/last/pretrained_model \
  . \
  --repo-type model
```

---

## Step 12: Local Evaluation (run on local machine)

Download checkpoint from HuggingFace:
```bash
cd /media/zihan-gao/lehome-challenge

huggingface-cli download <YOUR_HF_USERNAME>/<MODEL_REPO_NAME> \
  --repo-type model \
  --local-dir outputs/train/xvla_finetune_top_long_h100_v2/checkpoints/last/pretrained_model
```

Run evaluation:
```bash
conda activate leisaac_dev

python -m scripts.eval \
  --policy_type lerobot \
  --policy_path outputs/train/xvla_finetune_top_long_h100_v2/checkpoints/last/pretrained_model \
  --garment_type top_long \
  --dataset_root Datasets/example/top_long_merged \
  --num_episodes 5 \
  --enable_cameras \
  --task_description "fold the garment on the table" \
  --headless
```

---

## Resume Interrupted Training

```bash
# If training was interrupted, resume from last checkpoint:
lerobot-train --config_path=configs/train_xvla.yaml --resume=true
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `tmux new -s xvla` | New persistent session |
| `tmux attach -t xvla` | Re-attach after disconnect |
| `Ctrl+B D` | Detach (training keeps running) |
| `watch -n 5 nvidia-smi` | Monitor GPU |
| `lerobot-train --config_path=configs/train_xvla.yaml` | Start training |
| `lerobot-train ... --resume=true` | Resume training |

## Notes

- **ARM64 PyTorch**: Always use pip with `--index-url https://download.pytorch.org/whl/cu126`. Do NOT use `conda install pytorch` or `--index-url cu124` (no aarch64 CUDA wheels).
- **Checkpoint path for eval**: must point to `checkpoints/last/pretrained_model/` (not `checkpoints/last/`).
- **bfloat16 → numpy**: eval script requires `.float()` before `.numpy()` (already patched in `scripts/eval_policy/lerobot_policy.py:117`).
- **IsaacSim**: only runs locally, not on remote server.
