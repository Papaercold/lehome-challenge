#!/bin/bash
# =============================================================================
# SARM 奖励模型训练脚本
# 前置条件：已运行 sarm_annotate.sh 完成 Qwen VLM 标注
# 用法：bash sarm_train.sh
# =============================================================================

# ── 数据集（必须与 sarm_annotate.sh 中的 REPO_ID 一致）──────────────────────
REPO_ID="Papercold/top-long-merged"

# ── 训练步数（250条数据推荐 5000，数据量翻倍可加到 10000）───────────────────
STEPS=5000

# ── 输出目录（训练完的 SARM 模型保存在这里）──────────────────────────────────
OUTPUT_DIR="outputs/train/sarm_top_long"

# ── 是否推送到 HuggingFace（训练后方便在其他机器加载）──────────────────────
PUSH_TO_HUB=false          # true = 推送；false = 只保存本地

# =============================================================================
# 执行训练（一般不需要改下面的内容）
# =============================================================================
set -e

echo "=============================="
echo "SARM 奖励模型训练开始"
echo "  数据集:   $REPO_ID"
echo "  训练步数: $STEPS"
echo "  输出目录: $OUTPUT_DIR"
echo "=============================="

lerobot-train \
    --config_path configs/train_sarm.yaml \
    --dataset.repo_id "$REPO_ID" \
    --steps $STEPS \
    --output_dir "$OUTPUT_DIR" \
    --policy.push_to_hub $PUSH_TO_HUB

echo "=============================="
echo "SARM 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo ""
echo "下一步：运行 bash sarm_rabc.sh 计算 RA-BC 权重"
echo "=============================="
