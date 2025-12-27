#!/bin/bash
# ==========================================================
# 🚀 Fine-tune OLMo-2-1B-Instruct on radiology JSONL data using Accelerate
# ==========================================================

source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig_finetune

export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

#Choose between "Qwen/Qwen2.5-1.5B-Instruct" #or "meta-llama/Llama-3.2-1B-Instruct" or  ""allenai/OLMo-2-0425-1B-Instruct""
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
TRAIN_FILE="./dummy_datasets/jsonl/mimic_inspired_train_context_impression-finetune.jsonl"
OUTPUT_DIR="./outputs/llama_finetune/"
# -------------------------------
# 🪵 Robust log directory handling
# -------------------------------
LOG_DIR="./logs"   # safer: local directory instead of root
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +'%Y%m%d_%H%M')
LOGFILE="${LOG_DIR}/sft_radiology_${TIMESTAMP}.log"

echo "📁 Logs will be saved to: ${LOGFILE}"


echo "🚀 Starting Accelerate fine-tuning..."
echo "🧠 Model: ${MODEL_NAME}"
echo "📄 Train file: ${TRAIN_FILE}"
echo "💾 Output dir: ${OUTPUT_DIR}"
echo "🧾 Log file: ${LOGFILE}"

nohup accelerate launch --config-file default_config.yaml ./core_scripts/sft.py \
  --model_name_or_path ${MODEL_NAME} \
  --train_file ${TRAIN_FILE} \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --gradient_checkpointing \
  --output_dir ${OUTPUT_DIR} \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 5 \
  --logging_steps 100 \
  --bf16 True \
  --fp16 False \
  --tf32 True \
  > ${LOGFILE} 2>&1 &

echo "✅ Fine-tuning launched in background."
echo "📜 To monitor progress: tail -f ${LOGFILE}"
