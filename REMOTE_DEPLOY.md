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

## Step 5: Install lerobot and Dependencies

```bash
pip install "lerobot[xvla]"
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

## Step 6: Clone Repository

```bash
git clone <YOUR_REPO_URL> ~/lehome-challenge
cd ~/lehome-challenge
```

Or sync from local machine (run on **local**):
```bash
rsync -avz --progress \
  --exclude='outputs/' --exclude='Datasets/' --exclude='Assets/' \
  --exclude='third_party/' --exclude='__pycache__/' --exclude='.git/' \
  /media/zihan-gao/lehome-challenge/ \
  ubuntu@<SERVER_IP>:~/lehome-challenge/
```

---

## Step 7: Download Dataset

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

## Step 8: Start Training

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

## Step 9: Monitor Training

In a separate SSH window:
```bash
# GPU utilization
watch -n 5 nvidia-smi

# Training logs
tail -f ~/lehome-challenge/outputs/train/xvla_finetune_top_long_h100/train.log
```

Expected loss curve: starts ~0.7, drops to ~0.05–0.03 over 100K steps.

---

## Step 10: Download Checkpoint (run on local machine)

```bash
rsync -avz --progress \
  ubuntu@<SERVER_IP>:~/lehome-challenge/outputs/train/xvla_finetune_top_long_h100/checkpoints/last/pretrained_model/ \
  /media/zihan-gao/lehome-challenge/outputs/train/xvla_finetune_top_long_h100/checkpoints/last/pretrained_model/
```

---

## Step 11: Local Evaluation

```bash
cd /media/zihan-gao/lehome-challenge
conda activate leisaac_dev

python -m scripts.eval \
  --policy_type lerobot \
  --policy_path outputs/train/xvla_finetune_top_long_h100/checkpoints/last/pretrained_model \
  --garment_type "tops_long" \
  --dataset_root Datasets/example/top_long_merged \
  --num_episodes 5 \
  --enable_cameras \
  --task_description "fold the garment on the table"
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
