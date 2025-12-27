#!/bin/bash
set -e


# ===========================
# 🧠 Environment setup
# ===========================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig_instruct    # same env as save-datastore step

export CUDA_VISIBLE_DEVICES=4

LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logs → $LOG"

# Environment-level determinism

#export PYTHONHASHSEED=0
#export CUDA_LAUNCH_BLOCKING=1
#export CUBLAS_WORKSPACE_CONFIG=:16:8
#export CUDNN_DETERMINISTIC=1
#export CUDNN_BENCHMARK=0

# ===========================
# ⚙️ Configuration
# ===========================
export CUDA_VISIBLE_DEVICES=6

##############################################
# USER PARAMETERS
##############################################

# Repo containing knnlm.py + retomaton.py
REPO_PATH="./core_scripts"

# Models
#Choose between "Qwen/Qwen2.5-1.5B-Instruct" #or "meta-llama/Llama-3.2-1B-Instruct" or  ""allenai/OLMo-2-0425-1B-Instruct""
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
FINETUNED_MODEL="/data/nuri/retomaton_healthcare/outputs/llama_finetune/"

# Dataset
JSONL="./dummy_datasets/jsonl/mimic_inspired_test_context_impression-finetune.jsonl"

# Datastore
DSTORE_DIR=./datastores/dummy/meta-llama/Llama-3.2-1B-Instruct/
DSTORE_SIZE=281325
DIM=2048

# Retrieval params
K=1024
LAMBDA=0.5
KNN_TEMP=1.0
MIN_KNN=500
MAX_KNN=2048

# Generation hyperparameters
#TEMP=0.7
#TOPP=0.9

# Deterministic decoding
TEMP=0.0
TOPP=1.0

# Number of records to evaluate
LIMIT=60

##############################################
# OUTPUT DIRECTORY WITH FULL PARAM ENCODING
##############################################

OUTPUT_DIR="outputs/llama_k${K}_lambda${LAMBDA}_min${MIN_KNN}_max${MAX_KNN}_T${TEMP}_p${TOPP}"
mkdir -p "${OUTPUT_DIR}"

LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logs → $LOG"

##############################################
# RUN PIPELINE
##############################################

nohup python -u ./core_scripts/4_generations_perplexity_debug.py \
  --repo_path "${REPO_PATH}" \
  --model_name "${MODEL_NAME}" \
  --finetuned_dir "${FINETUNED_MODEL}" \
  --jsonl "${JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dstore_dir "${DSTORE_DIR}" \
  --dstore_size ${DSTORE_SIZE} \
  --dimension ${DIM} \
  --k ${K} \
  --lambda_val ${LAMBDA} \
  --knn_temp ${KNN_TEMP} \
  --min_knn ${MIN_KNN} \
  --max_knn ${MAX_KNN} \
  --temperature ${TEMP} \
  --top_p ${TOPP} \
  --limit ${LIMIT} \
  > "${LOG}" 2>&1 &

PID=$!
echo "Running as PID=${PID}"
echo "Monitor with:"
echo "  tail -f ${LOG}"
