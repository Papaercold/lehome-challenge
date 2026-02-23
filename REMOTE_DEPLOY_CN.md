# 远程训练部署指南

GH200（ARM64 aarch64）+ X-VLA 全量微调 LeHome Challenge 数据集。

---

## 前提条件

- 远程服务器：NVIDIA GH200 / H100（ARM64 aarch64 架构）
- 本地机器：任意系统（用于本地 IsaacSim 评估）
- 代码托管：GitHub
- 模型权重托管：HuggingFace Hub

---

## 第一步：SSH 连接远程服务器

```bash
ssh ubuntu@<服务器IP>
# 如果有密钥文件：
ssh -i ~/.ssh/your_key.pem ubuntu@<服务器IP>
```

---

## 第二步：安装 Miniforge（ARM64 版）

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
echo 'source ~/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc
```

---

## 第三步：创建 Python 环境

```bash
conda create -n xvla_train python=3.11 -y
conda activate xvla_train
```

---

## 第四步：安装 PyTorch（ARM64 必须用 pip，不能用 conda）

```bash
# CUDA 12.6 wheel，兼容 CUDA 12.6/12.8 驱动
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

验证：
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# 预期输出：CUDA: True 12.6 NVIDIA GH200 480GB
```

---

## 第五步：安装 lerobot 和依赖

```bash
pip install "lerobot[xvla]"
pip install "transformers>=4.47.0,<5.0.0"
pip install draccus huggingface_hub
```

> `transformers` 版本必须锁定：低于 4.47 不支持 Florence2LanguageConfig，高于 5.0 不兼容 huggingface-hub 1.x。

验证：
```bash
python -c "import lerobot; print('lerobot OK')"
python -c "from transformers import AutoModel; print('transformers OK')"
```

---

## 第六步：拉取代码

```bash
# 首次部署：克隆仓库
git clone <你的GitHub仓库地址> ~/lehome-challenge

# 后续更新：拉取最新代码
cd ~/lehome-challenge
git pull
```

---

## 第七步：下载数据集（在远程服务器执行）

```bash
cd ~/lehome-challenge

# 私有数据集需要先登录（在 huggingface.co/settings/tokens 生成 token）
huggingface-cli login

# 下载合并版数据集（无深度图，包含4种衣物类型）
huggingface-cli download lehome/dataset_challenge_merged \
  --repo-type dataset \
  --local-dir Datasets/example
```

验证：
```bash
ls Datasets/example/
# 预期看到：four_types_merged  pant_long_merged  pant_short_merged  top_long_merged  top_short_merged
```

> **训练不需要下载 Assets**，Assets 只用于本地 IsaacSim 仿真评估。

---

## 第八步：启动训练

```bash
cd ~/lehome-challenge
conda activate xvla_train

# 创建 tmux 会话，防止 SSH 断线后训练停止
tmux new -s xvla

# 启动训练（首次运行会自动从 HuggingFace 下载 xvla-base 约 3.5GB）
lerobot-train --config_path=configs/train_xvla.yaml
```

- 断开 SSH 但保持训练继续：`Ctrl+B` 然后按 `D`（detach）
- 重新连回查看日志：`tmux attach -t xvla`

---

## 第九步：监控训练

另开一个 SSH 窗口：
```bash
# 查看 GPU 使用情况
watch -n 5 nvidia-smi

# 查看训练日志
tail -f ~/lehome-challenge/outputs/train/xvla_finetune_top_long_h100/train.log
```

预期 loss 曲线：从约 0.7 开始，100K steps 后降至 0.03–0.05。

---

## 第十步：上传 Checkpoint 到 HuggingFace（在远程服务器执行）

```bash
cd ~/lehome-challenge
conda activate xvla_train

# 登录 HuggingFace（只需做一次）
huggingface-cli login

# 上传训练好的模型权重
huggingface-cli upload <你的HF用户名>/<模型仓库名> \
  outputs/train/xvla_finetune_top_long_h100/checkpoints/last/pretrained_model \
  --repo-type model
```

上传中间 checkpoint（可选）：
```bash
huggingface-cli upload <你的HF用户名>/<模型仓库名> \
  outputs/train/xvla_finetune_top_long_h100/checkpoints/010000/pretrained_model \
  --repo-type model \
  --commit-message "step 10000"
```

---

## 第十一步：本地评估（在本地机器执行）

先从 HuggingFace 下载 checkpoint：
```bash
cd /media/zihan-gao/lehome-challenge

huggingface-cli download <你的HF用户名>/<模型仓库名> \
  --repo-type model \
  --local-dir outputs/train/xvla_finetune_top_long_h100/checkpoints/last/pretrained_model
```

然后启动评估：
```bash
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

## 中断后恢复训练

```bash
# 从上次保存的 checkpoint 继续训练（每 10000 步自动保存一次）
lerobot-train --config_path=configs/train_xvla.yaml --resume=true
```

---

## 常用命令速查

| 命令 | 用途 |
|------|------|
| `tmux new -s xvla` | 新建持久会话 |
| `tmux attach -t xvla` | 断线后重新连接 |
| `Ctrl+B D` | 断开会话（训练继续跑） |
| `watch -n 5 nvidia-smi` | 监控 GPU |
| `lerobot-train --config_path=configs/train_xvla.yaml` | 启动训练 |
| `lerobot-train ... --resume=true` | 恢复训练 |

---

## 注意事项

- **ARM64 PyTorch 安装**：必须用 `pip install torch --index-url https://download.pytorch.org/whl/cu126`，不能用 `conda install pytorch`，也不能用 `cu124`（aarch64 没有对应的 CUDA wheel）。
- **eval 的 checkpoint 路径**：必须指向 `checkpoints/last/pretrained_model/`，不是 `checkpoints/last/`。
- **bfloat16 转 numpy**：eval 脚本需要在 `.numpy()` 前加 `.float()`，已在 `scripts/eval_policy/lerobot_policy.py` 第 117 行修复。
- **IsaacSim 仿真**：只能在本地运行，远程服务器不需要也无法运行。
