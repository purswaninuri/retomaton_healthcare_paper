#!/bin/bash
# ==========================================
# Convert train / val / test CSVs into JSONL fine-tune format
# using your Python converter script.
# ==========================================

# Exit immediately on error
set -e

# === CONFIGURATION ===
INPUT_DIR="./dummy_datasets/"
OUTPUT_DIR="./dummy_datasets/jsonl/"
PYTHON_SCRIPT="./core_scripts/convert_to_jsonl.py"

# === FILE NAMES (edit as needed) ===
TRAIN_FILE="mimic_inspired_train_context_impression.csv"
VAL_FILE="mimic_inspired_val_context_impression.csv"
TEST_FILE="mimic_inspired_test_context_impression.csv"

# === CREATE OUTPUT DIR ===
mkdir -p "$OUTPUT_DIR"

# === CONVERT EACH FILE ===
echo "🚀 Converting CSVs to fine-tune JSONL format..."
echo

for FILE in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
    echo "📄 Processing $FILE ..."
    python3 "$PYTHON_SCRIPT" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --files "$FILE" \
        --finetune
    echo
done

echo "✅ Conversion complete! JSONL files saved in: $OUTPUT_DIR"
