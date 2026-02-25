#!/bin/bash
# =============================================================================
# SARM 子任务标注脚本（Qwen3-VL 自动标注）
# 用法：bash scripts/sarm_annotate.sh
# =============================================================================

# ── 必填：你的 HuggingFace 数据集 repo ──────────────────────────────────────
REPO_ID="your-username/your-dataset-name"   # ← 改成你上传的 HF repo ID

# ── 子任务定义（逗号分隔，Qwen 会根据这些名字在视频里找分界点）────────────────
# 越具体越好，和实际动作顺序一致
DENSE_SUBTASKS="Position arms above garment,Grab corners of garment,Fold first half onto second half,Flatten and align edges,Complete fold"

# ── 使用的摄像头（用哪个视角让 Qwen 看）──────────────────────────────────────
VIDEO_KEY="observation.images.top_rgb"      # top / left / right 三选一，top 视野最全

# ── VLM 模型（默认 30B MoE，GH200 96GB 够用）────────────────────────────────
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# 如果显存不够可以改成更小的：
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"      # 7B，显存要求低，精度略差

# ── 并行 worker 数（单卡填 1，多卡填卡数）──────────────────────────────────────
NUM_WORKERS=1
# GPU_IDS="0 1"                             # 多卡时取消注释，改成实际 GPU ID

# ── 标注完成后是否推送回 HuggingFace ─────────────────────────────────────────
PUSH_TO_HUB="--push-to-hub"                 # 推送：保留此行；不推送：注释此行
# OUTPUT_REPO_ID=""                         # 推送到不同 repo 时填写，否则覆盖原 repo

# ── 可视化（标注完自动生成 N 个 episode 的时间轴图，0=跳过）──────────────────
NUM_VISUALIZATIONS=5
VIZ_OUTPUT_DIR="outputs/sarm_viz"

# =============================================================================
# 执行标注（一般不需要改下面的内容）
# =============================================================================
set -e

echo "=============================="
echo "SARM 标注开始"
echo "  数据集:   $REPO_ID"
echo "  子任务数: $(echo $DENSE_SUBTASKS | tr ',' '\n' | wc -l)"
echo "  VLM 模型: $MODEL"
echo "=============================="

python -m lerobot.data_processing.sarm_annotations.subtask_annotation \
    --repo-id "$REPO_ID" \
    --dense-only \
    --dense-subtasks "$DENSE_SUBTASKS" \
    --video-key "$VIDEO_KEY" \
    --model "$MODEL" \
    --num-workers $NUM_WORKERS \
    --num-visualizations $NUM_VISUALIZATIONS \
    --visualize-type dense \
    --output-dir "$VIZ_OUTPUT_DIR" \
    --skip-existing \
    $PUSH_TO_HUB

echo "=============================="
echo "标注完成！"
echo "可视化结果保存在: $VIZ_OUTPUT_DIR"
echo "=============================="
