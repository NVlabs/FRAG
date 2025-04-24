#!/bin/bash

DATASET=$1
num_frames=$2
CHUNKS=$3
IDX=$4

source "$(dirname "${BASH_SOURCE[0]}")"/path.sh

filename=$(basename "$0")
filename="${filename%.*}"
output_dir=${output_root}/${DATASET}/$SPLIT/${filename}_${num_frames}

python eval_model.py \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --doc-path $doc_path \
    --visual-folder $visual_path \
    --output-dir $output_dir \
    --model "internvl" \
    --model-path "${root}/ckpts/InternVL2-8B" \
    --sample_frames $num_frames \
    --input_frames 1 \
    --selector_method "annot_scores_frames" \
    --image-aspect-ratio 12 \
    --dataset $DATASET \
    --split $SPLIT \
    --main-process


