#!/bin/bash
# Test VLA bridge agent (heuristic fallback, no VLM server needed)
port=10010
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--experiment_name VLA_bridge \
--run_id vla_heuristic_1 \
--port $port \
--agents vla_bridge vla_bridge \
--max_tokens 256 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 1 2 \
--screen_size 256

pkill -f -9 "port $port"
