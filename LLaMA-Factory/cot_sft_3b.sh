WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,2,4,5 torchrun --master_port 65502 --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 222.29.51.9  src/train.py --deepspeed examples/deepspeed/ds_z3_config.json --stage sft  --do_train True --use_fast_tokenizer True --flash_attn 'auto' --model_name_or_path /data/hehulingxiao/code/ICCV/models/Qwen2.5-VL-3B-Instruct --dataset QA_CoT_sample2 --template qwen2_vl --output_dir /data/hehulingxiao/code/PAPO_rebuttal/checkpoints/iclr26/qwen2_5_vl_3b__stage1__sample2 --warmup_steps 6 --weight_decay 0.1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --ddp_timeout 9000 --learning_rate 5e-6 --lr_scheduler_type cosine --logging_steps 1 --cutoff_len 4096 --save_steps 1000 --num_train_epochs 10 --bf16 --finetuning_type full

#!/bin/bash

WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --master_port 65501 \
  --nproc_per_node 4 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr 222.29.51.9 \
  src/train.py \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --stage sft \
  --do_train True \
  --use_fast_tokenizer True \
  --flash_attn auto \
  --model_name_or_path /data/hehulingxiao/code/ICCV/models/Qwen2.5-VL-3B-Instruct \
  --dataset QA_CoT_sample2 \
  --template qwen2_vl \
  --output_dir ../checkpoints/Fine-R1-3B-Stage1 \
  --warmup_steps 6 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --ddp_timeout 9000 \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --cutoff_len 4096 \
  --save_steps 1000 \
  --num_train_epochs 10 \
  --bf16 \
  --finetuning_type full