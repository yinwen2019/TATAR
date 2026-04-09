#!/bin/bash
set -e
set -x

cd "$(dirname "$0")/../../"

MODEL_PATH=/chanxueyan/ano_people/savemodel/QandA_epoch6/checkpoint-1000
OUT_DIR=src/eval/results/QandA_epoch6/checkpoint-1000
GPUS=0,1,2,3
NUM_PROCESSES=4

export CUDA_VISIBLE_DEVICES=$GPUS
# DATASET： KonIQ KADID SPAQ ArtiMuse AVA FLICKR-AES TAD66K
for DATASET in KonIQ; do
  rm -f $OUT_DIR/${DATASET}.rank*.done $OUT_DIR/${DATASET}.rank*.jsonl $OUT_DIR/${DATASET}.metrics.json
  torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    src/eval/eval_uni_iqa_iaa.py \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --batch_size 4 \
    --out_dir $OUT_DIR
done