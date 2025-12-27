#!/bin/bash
# ============================================================
# 🚀 Run combined metric evaluation (ROUGE + BLEU + BERTScore + PPL)
# ============================================================

# 1️⃣ Load conda and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig_benchmarks

# 2️⃣ Select GPU (change this ID as needed)
export CUDA_VISIBLE_DEVICES=3

# 3️⃣ Define paths
INPUT_DIR=${1:-./outputs/llama_k1024_lambda0.5_min500_max2048_T0.0_p1.0}
SCRIPT_PATH="./core_scripts/5_all_metrics.py"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/metrics_eval_$(basename "$INPUT_DIR")_$(date +%Y%m%d_%H%M).log"

# 4️⃣ Launch job in background with nohup
nohup python -u "$SCRIPT_PATH" --input_dir "$INPUT_DIR" > "$LOG_FILE" 2>&1 &

# 5️⃣ Display status
PID=$!
echo "============================================================"
echo "✅ Running evaluate_metrics.py in background"
echo "🧠 Environment : neubig_benchmarks"
echo "🖥️  GPU Device  : $CUDA_VISIBLE_DEVICES"
echo "📂 Input Dir    : $INPUT_DIR"
echo "📜 Log file     : $LOG_FILE"
echo "🔢 PID          : $PID"
echo "============================================================"
echo "Monitor progress with:"
echo "   tail -f $LOG_FILE"
