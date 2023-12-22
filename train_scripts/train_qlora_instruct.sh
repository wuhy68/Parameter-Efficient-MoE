#!/bin/bash
master_addr=$1
nnodes=$2
node_rank=$3
master_addr=$1
nnodes=$2
node_rank=$3
pretrained_path=
data_path=
output_path=

torchrun --nproc_per_node=8 --nnodes=${nnodes} --master_addr ${master_addr} --master_port=11111 --node_rank ${node_rank} train_qlora.py \
    --model_name_or_path $pretrained_path \
    --data_path $data_path \
    --bf16 True \
    --model_max_length 512 \
    --output_dir $output_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate 2e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_steps 200 \
    --tf32 True