#!/bin/bash
# Start the script in its own process group
set -m


COMMON_ARGS="--batch_size 50 --output_directory YOUR-OUTPUT-PATH"

(
CUDA_VISIBLE_DEVICES=0 python run_mmpi_biased.py $COMMON_ARGS --model_name "Qwen/Qwen2-72B-Instruct" &
 wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."