#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-1} --master_port=$((RANDOM + 10000)) \
    $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3} 
