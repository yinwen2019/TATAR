set -ex

RUN_NAME="SFT_QandA_epoch2_step40"

export DEBUG_MODE="false"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

export FILELOCK_USE_SOFT=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# dist
nproc_per_node=${ARNOLD_WORKER_GPU:-8}

if [ -n "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  nnodes=1
  node_rank=0
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM:-1}
  node_rank=${ARNOLD_ID:-0}
  master_addr_var="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_port_var="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_addr=${!master_addr_var:-127.0.0.1}
  master_port=${!master_port_var:-12345}
  master_port=$(echo "$master_port" | tr ',' ' ' | awk '{print $1}')
fi

MODEL="/chanxueyan/ano_people/premodel/Qwen2.5-VL-7B-Instruct"
OUT="/chanxueyan/ano_people/savemodel/$RUN_NAME"

IQA_JSONL="/chanxueyan/ano_people/datasets/train_reversal_instruct_judged.jsonl"
IAA_JSONL="/chanxueyan/ano_people/datasets/train_cot_narrative_judged.jsonl"
IMG_ROOT="/chanxueyan/ano_people/datasets/"

torchrun --nproc_per_node="$nproc_per_node" \
  --nnodes="$nnodes" \
  --node_rank="$node_rank" \
  --master_addr="$master_addr" \
  --master_port="$master_port" \
  sft.py \
  --deepspeed   "../../local_scripts/zero3.json" \
  --output_dir "$OUT" \
  --model_name_or_path "$MODEL" \
  --iqa_jsonl "$IQA_JSONL" \
  --iaa_jsonl "$IAA_JSONL" \
  --image_root "$IMG_ROOT" \
  --mix_prob_iqa 0.5 \
  --max_pixels 1572864 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 2 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 40 \
  --save_only_model true \
  --bf16 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --report_to wandb \
  --run_name "$RUN_NAME"