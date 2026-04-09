set -x

export DEBUG_MODE="false"
RUN_NAME="QandA_guassian_Rank"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src


# set dist args
# SINGLE=1

nproc_per_node=${ARNOLD_WORKER_GPU:-8}


if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=8
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM:-1}
  node_rank=${ARNOLD_ID:-0}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr:-127.0.0.1}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port:-12345}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"


# set up envs
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --nproc_per_node=${nproc_per_node} \
    --nnodes=${nnodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    src/open_r1/uni_iqa_iaa.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir /chanxueyan/ano_people/savemodel/$RUN_NAME \
    --model_name_or_path /chanxueyan/ano_people/savemodel/SFT_QandA_epoch2_step40/checkpoint-120 \
    --dataset_iqa data_config/iqa_score.yaml \
    --dataset_iaa data_config/iaa_score.yaml \
    --image_root /chanxueyan/ano_people/datasets/ \
    --max_prompt_length 2048 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --score_reward_threshold 8.75 \
    --beta 0.001 \
    --max_pixels 1572864

