# export CUDA_VISIBLE_DEVICES=3

# for 9B
# DEVICES_IDS="GPU-a25958d0-9489-9c70-b235-06356c9525b7"
# MODEL_PATH="/ML-A100/team/sshare/home/xujianbo/github/LLaMA-Factory/output/sft_output/KY-9B-4K-g256-pack-0525/model"
# PARALLEL_SIZE=1

# for 34B
DEVICES_IDS='"GPU-a25958d0-9489-9c70-b235-06356c9525b7,GPU-3ec28546-b44d-17f9-70a9-715d01d437e2,GPU-2bd8e675-f006-6c66-c72b-40814df25a97,GPU-0421b225-9577-032b-6cbf-006407afc886"'
# MODEL_PATH="/ML-A100/team/sshare/home/xujianbo/github/LLaMA-Factory/output/sft_output/KY-34B-4K-g256-h-0620/model"
# MODEL_PATH="/ML-A100/team/sshare/home/xujianbo/github/LLaMA-Factory/output/sft_output/KY-34B-4K-g256-0531/model"
# MODEL_PATH="/ML-A100/team/sshare/home/xujianbo/github/LLaMA-Factory/output/sft_output/KY-34B-4K-g256-h-0619/model"

MODEL_PATH="/ML-A100/team/sshare/models/Higgs-Llama-3-70B"

PARALLEL_SIZE=4

# sudo docker run --rm -d --gpus "device=${DEVICES_IDS}"\
# sudo docker run --rm -d --gpus '"device=GPU-a25958d0-9489-9c70-b235-06356c9525b7,GPU-3ec28546-b44d-17f9-70a9-715d01d437e2"' \
# sudo docker run -d --gpus '"device=GPU-a25958d0-9489-9c70-b235-06356c9525b7,GPU-3ec28546-b44d-17f9-70a9-715d01d437e2"' \
sudo docker run -d --gpus '"device=GPU-a25958d0-9489-9c70-b235-06356c9525b7,GPU-3ec28546-b44d-17f9-70a9-715d01d437e2,GPU-2bd8e675-f006-6c66-c72b-40814df25a97,GPU-0421b225-9577-032b-6cbf-006407afc886"' \
	-v  /mnt/vepfs-cnsh3bbeb49a2456:/ML-A100  \
	--shm-size "100g" \
	-p 8208:8000 \
	--name Higgs-llama-3-70B \
    vllm/vllm-openai:latest \
	--model ${MODEL_PATH} \
	--served-model-name KY-9B-Chat-16k-0524 \
    --host 0.0.0.0 \
	--port 8000 \
	--gpu-memory-utilization 0.90 \
	--tokenizer-mode slow \
	--tensor-parallel-size $PARALLEL_SIZE
