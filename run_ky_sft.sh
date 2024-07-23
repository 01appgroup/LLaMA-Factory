# cd /ML-A100/team/align/home/chenjianqun/LLaMA-Factory
# pip install wandb -i https://mirrors.aliyun.com/pypi/simple/

set -x -e

export OMP_NUM_THREADS=1

# MODEL_PATH="/ML-A100/public/run/to_sft/Yi-V1.5-32K/Yi-34B-32K-v2_hf"
MODEL_PATH="/ML-A100/team/sshare/models/for_sft/Yi-34B-32K-v2_hf"
# MODEL_PATH="/ML-A100/team/sshare/data/kaiying/sft_output/sft_output/KY-34B-4K-g64-s1-0629-2/model"

PROJECT_NAME="TMP"
GLOBAL_BATCH_SIZE=256
BATCH_SIZE_PER_GPU=1
MODEL_NAME="KY-34B-4K-g${GLOBAL_BATCH_SIZE}-h-0723"

OUTPUT_PATH="output/sft_output/${MODEL_NAME}"

DATASET_NAME="ky_34b_sft_half,ky_34b_sft_ext"
# DATASET_NAME="ky_34b_sft_base,ky_34b_sft_ext"

# export WANDB_API_KEY=cb6bcb2df698f249880cb013bcbc07f75f13a457
# export WANDB_PROJECT=$PROJECT_NAME
# export WANDB_RUN_NAME=$MODEL_NAME

MASTER_PORT=${MLP_WORKER_0_PORT:-12345}
MASTER_HOST=${MLP_WORKER_0_HOST:-127.0.0.1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NODE_NUM=${MLP_WORKER_NUM:-1}
GPU_NUM=${MLP_WORKER_GPU:-8}

GRADIENT_ACC_STEPS=$((GLOBAL_BATCH_SIZE/BATCH_SIZE_PER_GPU/GPU_NUM))

echo "Master Port: $MASTER_PORT"

torchrun --nproc_per_node $GPU_NUM \
 --master_addr $MASTER_HOST \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NODE_NUM \
src/train.py \
    --deepspeed configs/ds_zero3_cqia.conf \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --template yi \
    --flash_attn fa2 \
    --use_fast_tokenizer False \
    --dataset ${DATASET_NAME} \
    --finetuning_type full \
    --output_dir ${OUTPUT_PATH} \
    --overwrite_output_dir \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --num_train_epochs 3.0 \
    --cutoff_len 4096 \
    --warmup_ratio 0.05 \
    --preprocessing_num_workers 16 \
    --plot_loss \
    --bf16 \
    --report_to tensorboard 


    # --lr_scheduler_type cosine_with_min_lr \
    # --lr_scheduler_kwargs '{"min_lr": 3e-6}' \
