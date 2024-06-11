#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}

# echo "CUDA_VISIBLE_DEVICES=$1 python main.py ${PY_ARGS}"
CUDA_VISIBLE_DEVICES=$1 python main.py ${PY_ARGS}
