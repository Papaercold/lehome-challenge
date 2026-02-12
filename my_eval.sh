#!/bin/bash
POLICY_PATH="outputs/train/act_top_long/checkpoints/last/pretrained_model"
GARMENT_TYPE="tops_long"
DATASET_ROOT="Datasets/example/top_long_merged"
SAVE_DIR="outputs/eval/act_top_long/eval_videos"
NUM_EPISODES=2

echo "开始执行 ACT 策略评估..."
echo "策略路径: $POLICY_PATH"
echo "服装类型: $GARMENT_TYPE"

# 执行 Python 脚本
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path "$POLICY_PATH" \
    --garment_type "$GARMENT_TYPE" \
    --dataset_root "$DATASET_ROOT" \
    --num_episodes $NUM_EPISODES \
    --enable_cameras \
    --video_dir "$SAVE_DIR" \
    --save_datasets \
    --device cpu

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "评估任务成功完成！"
else
    echo "评估过程中出现错误。"
    exit 1
fi