#!/bin/bash

DATASET=$1
num_frames=$2

source "$(dirname "${BASH_SOURCE[0]}")"/path.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

filename=$(basename "$0")
filename="${filename%.*}"
output_dir=${output_root}/${DATASET}/$SPLIT/${filename}_${num_frames}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval_model.py \
        --doc-path $doc_path \
        --visual-folder $visual_path \
        --output-dir $output_dir \
        --model "internvl" \
        --model-path "${root}/ckpts/InternVL2-8B" \
        --sample_frames $num_frames \
        --input_frames 32 \
        --selector_method "topk_frames" \
        --score-docs "${output_root}/${DATASET}/$SPLIT/annot_scores_internvl-8b_frames_256.json" \
        --image-aspect-ratio 1 \
        --dataset $DATASET \
        --split $SPLIT \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

python collect_results.py \
    --result-path $output_dir \
    --doc-path $doc_path \
    --dataset $DATASET \
    --split $SPLIT \
    --eval
