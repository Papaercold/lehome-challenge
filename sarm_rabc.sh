#!/bin/bash
# =============================================================================
# SARM RA-BC 权重计算脚本
# 前置条件：已运行 sarm_train.sh 完成 SARM 奖励模型训练
# 用法：bash sarm_rabc.sh
# =============================================================================

# ── 数据集（必须与标注和训练时一致）──────────────────────────────────────────
REPO_ID="Papercold/top-long-merged"

# ── SARM 模型路径（sarm_train.sh 的 OUTPUT_DIR/checkpoints/last）───────────
SARM_MODEL_PATH="outputs/train/sarm_top_long/checkpoints/last/pretrained_model"

# ── 使用哪个 SARM head 计算进度分数 ──────────────────────────────────────────
# dense：细粒度子任务 head（推荐，因为我们用了 dense_only 模式标注）
# sparse：粗粒度任务 head
HEAD_MODE="dense"

# ── 计算帧的步长（stride=1 每帧都算，stride=5 每5帧算一次再插值，更快）────
# 250条数据建议 stride=3 (速度/精度平衡)，极快可用 stride=5
STRIDE=3

# ── 可视化几条 episode（0=跳过，建议 5 用于检验 SARM 打分是否合理）──────────
NUM_VISUALIZATIONS=5
VIZ_OUTPUT_DIR="outputs/sarm_rabc_viz"

# ── 计算结果是否推送到 HuggingFace（训练时用 hf:// 路径加载）───────────────
# true = 推送，训练 X-VLA 时 rabc_progress_path 用 hf://datasets/... 格式
# false = 只保存本地，训练时用本地绝对路径
PUSH_TO_HUB=true

# =============================================================================
# 执行计算（一般不需要改下面的内容）
# =============================================================================
set -e

echo "=============================="
echo "RA-BC 进度权重计算开始"
echo "  数据集:   $REPO_ID"
echo "  SARM 模型: $SARM_MODEL_PATH"
echo "  Head 模式: $HEAD_MODE"
echo "  计算步长:  stride=$STRIDE"
echo "=============================="

PUSH_FLAG=""
if [ "$PUSH_TO_HUB" = "true" ]; then
    PUSH_FLAG="--push-to-hub"
fi

python -m lerobot.policies.sarm.compute_rabc_weights \
    --dataset-repo-id "$REPO_ID" \
    --reward-model-path "$SARM_MODEL_PATH" \
    --head-mode "$HEAD_MODE" \
    --stride $STRIDE \
    --num-visualizations $NUM_VISUALIZATIONS \
    --output-dir "$VIZ_OUTPUT_DIR" \
    $PUSH_FLAG

echo "=============================="
echo "RA-BC 权重计算完成！"
echo "可视化结果保存在: $VIZ_OUTPUT_DIR"
echo ""
echo "下一步：在 configs/train_xvla.yaml 中添加以下配置，然后运行训练："
echo ""
echo "  use_rabc: true"
if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "  rabc_progress_path: hf://datasets/$REPO_ID/sarm_progress.parquet"
else
    DATASET_CACHE=$(python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('$REPO_ID', download_videos=False)
print(ds.root)
" 2>/dev/null || echo "<dataset_cache_dir>")
    echo "  rabc_progress_path: $DATASET_CACHE/sarm_progress.parquet"
fi
echo "  rabc_head_mode: $HEAD_MODE"
echo "  rabc_kappa: 0.01"
echo "=============================="
