#!/bin/bash

MODEL_PATH="../checkpoints/Fine-R1-3B"
PROMPT_DIR="./data"
OUTPUT_DIR="../results"

DATASETS=("aircraft" "bird" "car" "dog" "flower" "pet")

BATCH_SIZE=2
DEVICE=${DEVICE:-"cuda:0"}
MAX_NEW_TOKENS=2048

mkdir -p $OUTPUT_DIR

echo "Starting Closed-World Evaluation..."
echo "Model: $MODEL_PATH"

for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "seen" "unseen"; do

        PROMPT_FILE="${PROMPT_DIR}/${DATASET}_${SPLIT}.jsonl"

        if [ ! -f "$PROMPT_FILE" ]; then
            echo "Missing file: $PROMPT_FILE — skipped."
            continue
        fi

        python evaluation.py \
            --mode closed \
            --model_path $MODEL_PATH \
            --prompt_path "$PROMPT_FILE" \
            --output_path "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --max_new_tokens $MAX_NEW_TOKENS

    done
done

echo "Closed-World Evaluation finished."
