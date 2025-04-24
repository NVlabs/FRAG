# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FRAG/blob/main/LICENSE

import argparse
import os
import json
import numpy as np

from tasks.utils import load_video, load_images

from models.builder import build_model
from frame_selection import FrameSelection
from collections import defaultdict


def main(args):
    generation_args = {"max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature,
                       "do_sample": args.do_sample}

    # answering LMM
    model_path = os.path.expanduser(args.model_path)
    model = build_model(args.model,
                        model_path,
                        generation_args,
                        image_aspect_ratio=args.image_aspect_ratio)

    # scroing LMM
    if args.selector_model is None:
        selector_model = model
    else:
        selector_model_path = os.path.expanduser(args.selector_model_path)
        selector_model = build_model(args.selector_model,
                                     args.selector_model_path,
                                     generation_args,
                                     image_aspect_ratio=args.selector_image_aspect_ratio)

    selector = FrameSelection(selector_model, args.selector_method, args.input_frames, args.sample_frames)

    # load input
    if args.input_type == 'video':
        images, frame_indices = load_video(args.input_path, args.sample_frames)
    elif args.input_type == 'images':
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        image_files = [file for file in os.listdir(args.input_path) if file.lower().endswith(image_extensions)]
        image_files = sorted(image_files)
        image_files = [os.path.join(args.input_path, file) for file in image_files]
        images, frame_indices = load_images(image_files, args.sample_frames)
    else:
        raise NotImplementedError(f"unknown input type: {args.input_type}")

    doc = {}
    doc['visual_path'] = args.input_path
    doc['question'] = args.query

    prompt = doc["question"].strip()
    if args.pre_prompt is not None:
        prompt = f"{args.pre_prompt}{prompt}"
    if args.post_prompt is not None:
        prompt = f"{prompt}{args.post_prompt}"
    doc['prompt'] = prompt

    selected_images_list, selected_indices_list, scores_dict = selector.select_frames(doc, images, frame_indices, return_scores=True)
    selected_images = selected_images_list[0]
    selected_indices = selected_indices_list[0]
    doc["selected_frames"] = selected_indices
    doc["scores"] = scores_dict

    outputs = model.run_model(selected_images, (prompt, None))
    doc["pred"] = outputs
    print(f"FRAG: {outputs}")

    if isinstance(next(iter(doc["scores"])), tuple):
        doc["scores"] = {str(k): v for k, v in doc["scores"].items()}

    out_path = os.path.join(args.output_dir, "%s.json" % os.path.basename(doc['visual_path']))
    with open(out_path, "w") as f:
        json.dump(doc, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs")

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--model", type=str, default="llava_ov")
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--image-aspect-ratio", type=str, default="anyres_max_9")
    parser.add_argument("--selector-model", type=str, default=None)
    parser.add_argument("--selector-model-path", type=str, default=None)
    parser.add_argument("--selector-image-aspect-ratio", type=str, default="anyres_max_9")

    parser.add_argument("--sample_frames", type=int, default=64)
    parser.add_argument("--input_frames", type=int, default=1)
    parser.add_argument("--selector_method", type=str, default="topk_frames")

    parser.add_argument("--input-type", type=str, default="video")
    parser.add_argument("--input-path", type=str, default="")

    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--pre-prompt", type=str, default=None)
    parser.add_argument("--post-prompt", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.do_sample = args.temperature > 0

    main(args)

