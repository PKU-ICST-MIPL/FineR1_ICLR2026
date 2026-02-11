#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=4,5,6,7
N_GPU=4

MODEL_PATH="/data/hehulingxiao/code/PAPO/checkpoints/iclr26/qwen2_5_vl_7b__stage1" #"checkpoints/Fine-R1-7B-Stage1"

TOTAL_EPOCHES=5 
GLOBAL_BATCH_SIZE=64 
ROLLOUT_BATCH_SIZE=192 
MINI_ROLLOUT_BATCH_SIZE=64
VAL_BATCH_SIZE=256
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
MAX_PIXELS=1003520
SAVE_FREQ=5
SAVE_LIMIT=3

EXP_NAME="Fine-R1-7B-Stage2"

CONGI_FILE="examples/configs/config_tapo.yaml"
TRAIN_FILE="data/Fine-R1-Stage2-data/data"

FORMAT_PROMPT="examples/format_prompt/cls_thinking.jinja"
REWARD_FUNCTION="examples/reward_function/cls_thinking.py:compute_score"

KL_PRCP_COEF=0.01
USE_AUG_ENTROPY_LOSS=true
AUG_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.03

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.is_noisy=true \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    algorithm.use_aug_entropy_loss=${USE_AUG_ENTROPY_LOSS} \
    algorithm.aug_entropy_loss_coef=${AUG_ENTROPY_LOSS_COEF} \
    algorithm.use_ori_entropy_loss=${USE_ORI_ENTROPY_LOSS} \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF}
