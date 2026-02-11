#!/bin/bash

MODEL_PATH="./checkpoints/Fine-R1-3B"
PROMPT_DIR="./data/eval"
OUTPUT_DIR="./results"

DATASETS=("aircraft" "bird" "car" "dog" "flower" "pet")

BATCH_SIZE=2
DEVICE=${DEVICE:-"cuda:1"}
MAX_NEW_TOKENS=2048

mkdir -p $OUTPUT_DIR

echo "Starting Open-World Evaluation..."
echo "Model: $MODEL_PATH"

for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "seen" "unseen"; do

        PROMPT_FILE="${PROMPT_DIR}/${DATASET}_${SPLIT}.jsonl"

        if [ ! -f "$PROMPT_FILE" ]; then
            echo "Missing file: $PROMPT_FILE — skipped."
            continue
        fi

        python evaluation.py \
            --mode open \
            --model_path $MODEL_PATH \
            --prompt_path "$PROMPT_FILE" \
            --output_path "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --siglip_path "google/siglip-base-patch16-256"

    done
done

echo "Evaluation finished."
