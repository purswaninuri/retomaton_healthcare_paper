#!/bin/bash

# ===========================
# 🧠 Environment setup
# ===========================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig_instruct    # ← change this to your env name if different

# ===========================
# ⚙️ Configuration
# ===========================
export CUDA_VISIBLE_DEVICES=5
#Choose between "Qwen/Qwen2.5-1.5B-Instruct" #or "meta-llama/Llama-3.2-1B-Instruct" or  ""allenai/OLMo-2-0425-1B-Instruct""
export MODEL="meta-llama/Llama-3.2-1B-Instruct" 
export SCRIPT="./core_scripts/run_clm_chat.py"
export TRAIN_FILE="./dummy_datasets/jsonl/mimic_inspired_train_context_impression-finetune.jsonl"
export VAL_FILE="./dummy_datasets/jsonl/mimic_inspired_val_context_impression-finetune.jsonl"
export OUT_DIR="outputs/${MODEL}"
export DSTORE_DIR="datastores/dummy/${MODEL}"

# ===========================
# 🪵 Logging setup
# ===========================
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/eval_knnlm_${TIMESTAMP}.log"

# ===========================
# 🚀 Run
# ===========================
echo "[INFO] Starting evaluation with KNN-LM datastore..."
nohup python -u ${SCRIPT} \
  --model_name_or_path ${MODEL} \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VAL_FILE} \
  --do_eval --eval_subset train \
  --block_size 1024 \
  --stride 1024 \
  --per_device_eval_batch_size 1 \
  --output_dir ${OUT_DIR} \
  --dstore_dir ${DSTORE_DIR} \
  --apply_chat_template \
  --save_knnlm_dstore \
  > ${LOG_FILE} 2>&1 &

echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Process started in background with PID $!"