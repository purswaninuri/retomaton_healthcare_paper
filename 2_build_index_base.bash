#!/bin/bash
# ============================================================
# 🌙 Memory-Efficient FAISS Index Build (tmux-safe)
# For: Qwen2.5-1.5B-Instruct datastore
# ============================================================

set -e  # stop on error
set -o pipefail

# ===========================
# 🧠 Environment Setup
# ============================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neubig_instruct

# Safety check
if [ -z "$CONDA_DEFAULT_ENV" ]; then
  echo "[ERROR] Conda environment not activated."
  exit 1
fi

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

# ---- FAISS INDEX PARAMS (memory-optimized IVFPQ) ----
export DSTORE_SIZE=281325     # ~49M keys from your earlier run
export NCENTROIDS=4096             # fewer centroids = lower RAM
export CODE_SIZE=64                # PQ code size
export PROBE=8                     # lower probe = MUCH lower RAM
export NUM_KEYS_TO_ADD=50000      # add keys in small chunks to avoid RAM spikes

# ---- Memory Controls ----
export MOVE_DSTORE_TO_MEM=false    # NEVER load 50M keys into RAM
export NO_LOAD_KEYS=true           # avoids loading keys.npy into RAM
export RECOMPUTE_DISTS=false       # keeps GPU overhead minimal

# ===========================
# 🪵 Logging Setup
# ===========================
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/build_index_${TS}.log"

echo "[INFO] Writing logs to: ${LOG_FILE}"

# ===========================
# 🚀 Run (no nohup; tmux handles persistence)
# ===========================
echo "[INFO] Starting FAISS index build…"
echo "[INFO] Running inside tmux is recommended."

nohup python -u "${SCRIPT}" \
  --model_name_or_path "${MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VAL_FILE}" \
  --apply_chat_template \
  --dstore_dir "${DSTORE_DIR}" \
  --dstore_size "${DSTORE_SIZE}" \
  --build_index \
  --ncentroids "${NCENTROIDS}" \
  --code_size "${CODE_SIZE}" \
  --probe "${PROBE}" \
  --num_keys_to_add_at_a_time "${NUM_KEYS_TO_ADD}" \
  --move_dstore_to_mem "${MOVE_DSTORE_TO_MEM}" \
  --no_load_keys "${NO_LOAD_KEYS}" \
  --recompute_dists "${RECOMPUTE_DISTS}" \
  --output_dir "${OUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &


echo "[INFO] Finished. Check log: ${LOG_FILE}"
