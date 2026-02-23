# Project: LeHome Challenge — X-VLA Training

## 项目概述

- **任务**：用 X-VLA 全量微调衣物折叠任务（bimanual SO-101 机器人）
- **远程服务器**：NVIDIA GH200（ARM64 aarch64），CUDA 12.6/12.8
- **本地**：IsaacSim 仿真评估（leisaac_dev conda 环境）
- **远程**：lerobot 训练（xvla_train conda 环境）

---

## 关键修复：夹爪训练 Bug（必须先做）

### 问题

lerobot 的 `so101_bimanual` action mode 在训练时强制把夹爪维度（dim 5, dim 11）归零，
导致模型永远无法学会闭合夹爪。

### 需要修改的文件

```
~/lerobot/src/lerobot/policies/xvla/action_hub.py
```

（如果是源码安装，路径在 `~/lerobot/` 下；如果是 pip 安装，用下面命令找路径）

```bash
python -c "import lerobot.policies.xvla.action_hub as m; import inspect; print(inspect.getfile(m))"
```

### 需要删除的 5 行（在 `BimanualSO101ActionSpace` 类里）

**`preprocess` 方法里删 3 行：**

```python
# 删除这行：
proprio_m[..., self.gripper_idx] = 0.0
# 删除这行（因为 body 被删了，if 留着会报语法错误）：
if action_m is not None:
# 删除这行：
    action_m[..., self.gripper_idx] = 0.0
```

**`postprocess` 方法里删 2 行：**

```python
# 删除这行：
if action.size(-1) > max(self.gripper_idx):
# 删除这行：
    action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
```

### 修改后 preprocess 应该是这样：

```python
def preprocess(self, proprio, action, mode="train"):
    proprio_m = self._pad_to_model_dim(proprio.clone())
    action_m = self._pad_to_model_dim(action.clone()) if action is not None else None
    return proprio_m, action_m
```

### 修改后 postprocess 应该是这样：

```python
def postprocess(self, action: torch.Tensor) -> torch.Tensor:
    if action.size(-1) < self.REAL_DIM:
        raise ValueError(f"Expected at least {self.REAL_DIM} dims in action, got {action.size(-1)}")
    return self._trim_to_real_dim(action)
```

### 自动 patch 脚本（推荐）

```bash
ACTION_HUB=$(python -c "import lerobot.policies.xvla.action_hub as m; import inspect; print(inspect.getfile(m))")
echo "Patching: $ACTION_HUB"

python -c "
path = '$ACTION_HUB'
with open(path) as f:
    lines = f.readlines()

out = []
for i, line in enumerate(lines):
    if 'gripper_idx] = 0.0' in line:
        continue
    if 'if action_m is not None:' in line and i+1 < len(lines) and 'gripper_idx] = 0.0' in lines[i+1]:
        continue
    if 'gripper_idx] = torch.sigmoid' in line:
        continue
    if 'if action.size(-1) > max(self.gripper_idx)' in line and i+1 < len(lines) and 'torch.sigmoid' in lines[i+1]:
        continue
    out.append(line)

with open(path, 'w') as f:
    f.writelines(out)
print('夹爪修复完成:', path)
"

# 验证（无输出即为成功）
grep -n "gripper_idx] = 0.0\|torch.sigmoid" $ACTION_HUB
```

---

## 训练配置

- **配置文件**：`configs/train_xvla.yaml`
- **action_mode**：`so101_bimanual`（12D 真实 → 20D 模型内部，补 8 个零）
- **有效维度**：修复前 10D（无夹爪），修复后 12D（含夹爪）
- **输出目录**：`outputs/train/xvla_finetune_top_long_h100/`
- **总步数**：100K steps，每 10K 保存一次
- **训练命令**：`lerobot-train --config_path=configs/train_xvla.yaml`

---

## 环境说明

- **conda 环境**：`xvla_train`
- **lerobot 安装方式**：源码可编辑安装（`pip install -e ".[xvla]"`），位于 `~/lerobot/`
- **lerobot 版本**：v0.4.3
- **PyTorch**：必须用 `pip install torch --index-url https://download.pytorch.org/whl/cu126`（ARM64 专用）

---

## 注意事项

- 修复夹爪 bug 后必须**从头重新训练**（不能 resume 旧 checkpoint，因为旧权重已学到夹爪=0）
- `transformers` 版本必须锁定在 `>=4.47.0,<5.0.0`
- checkpoint 上传到 HuggingFace 后在本地下载评估（IsaacSim 只能本地运行）
- 评估命令参数：`--garment_type top_long`（不是 `tops_long`）

---

## 完整部署流程

参见 `REMOTE_DEPLOY_CN.md`（中文）或 `REMOTE_DEPLOY.md`（英文）。
