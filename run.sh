#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python train.py --config configs/increfa.yml
