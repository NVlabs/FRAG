# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FRAG/blob/main/LICENSE

import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import copy

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from decord import VideoReader, cpu

from utils import split_list, get_chunk
from models.builder import build_model
from frame_selection import FrameSelection

from tasks.builder import build_task


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, task, video_docs, visual_folder, num_frames):
        self.task = task
        self.video_docs = video_docs
        self.video_names = list(video_docs.keys())
        self.visual_folder = visual_folder
        self.num_frames = num_frames

    def __getitem__(self, index):
        video_name = self.video_names[index]

        # load video
        video_path = os.path.join(self.visual_folder, video_name)
        doc = self.video_docs[video_name][0]
        if 'images' in doc:
            video_path = [os.path.join(self.visual_folder, x) for x in doc["images"]]
        images, frame_indices = self.task.load_visual(video_path, self.num_frames)

        docs = self.video_docs[video_name]
        prompts = []
        for doc in docs:
            prompt = self.task.doc_to_prompt(doc)
            prompts.append(prompt)

        return docs, prompts, images, frame_indices

    def __len__(self):
        return len(self.video_names)


def collate_fn(batch):
    docs, prompts, images, frame_indices = zip(*batch)
    docs = list(docs)
    prompts = list(prompts)
    images = list(images)
    frame_indices = list(frame_indices)
    return docs, prompts, images, frame_indices


# DataLoader
def create_data_loader(task, video_docs, visual_folder, num_frames, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(task, video_docs, visual_folder, num_frames)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # task
    task = build_task(args.dataset, args.split,
                      subtitles=args.subtitles,
                      visual_folder=args.visual_folder)

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

    selector = FrameSelection(selector_model, args.selector_method, args.input_frames, args.sample_frames,
                              score_docs=args.score_docs, sub_vid_cache=args.sub_vid_cache)

    docs = json.load(open(args.doc_path, "r"))

    # set id if doesn't exist
    for i, doc in enumerate(docs):
        if 'id' not in doc:
            doc["id"] = "%06d" % i
    
    # split by video
    video_docs = defaultdict(list)
    for doc in docs:
        video_name = task.doc_to_visual_name(doc)
        video_docs[video_name].append(doc)

    # chunk by videos
    video_names = sorted(list(video_docs.keys()))
    video_names = get_chunk(video_names, args.num_chunks, args.chunk_idx)
    video_docs_chunk = {}
    for video_name in video_names:
        video_docs_chunk[video_name] = video_docs[video_name]
    video_docs = video_docs_chunk

    # remove ones that are already done
    remaining_docs = defaultdict(list)
    for video_name in video_docs.keys():
        vid_docs = video_docs[video_name]
        for doc in vid_docs:
            out_name = os.path.join(args.output_dir, "%s.json" % doc["id"])
            if not os.path.exists(out_name):
                remaining_docs[video_name].append(doc)
    video_docs = remaining_docs

    data_loader = create_data_loader(task, video_docs, args.visual_folder, args.sample_frames)
    for docs, prompts, images, frame_indices in tqdm(data_loader, total=len(data_loader)):
        docs = docs[0]
        prompts = prompts[0]
        images = images[0]
        frame_indices = frame_indices[0]
        for doc, prompt in zip(docs, prompts):
            out_name = os.path.join(args.output_dir, "%s.json" % doc["id"])
            if os.path.exists(out_name):
                continue

            if args.annot_scores:
                scores, proposal_indices = selector.annotate_scores(doc, images, frame_indices)
                
                if scores is not None:
                    out = copy.deepcopy(doc)
                    out["frames"] = proposal_indices
                    out["scores"] = [float(x) for x in scores]
                    assert len(out["frames"]) == len(out["scores"])

                    with open(out_name, "w") as f:
                        json.dump(out, f)
            else:
                selected_images_list, selected_indices_list = selector.select_frames(doc, images, frame_indices)
                
                results = []
                for selected_images, selected_indices in zip(selected_images_list, selected_indices_list):
                    outputs = model.run_model(selected_images, prompt)
                    results.append(task.process_results(doc, outputs))

                out = copy.deepcopy(doc)
                out["frames"] = selected_indices_list
                out[task.result_key] = results
                assert len(out["frames"]) == len(out[task.result_key])

                with open(out_name, "w") as f:
                    json.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--visual-folder", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")    

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--model", type=str, default="llava_ov")
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--image-aspect-ratio", type=str, default="anyres_max_9")
    parser.add_argument("--selector-model", type=str, default=None)
    parser.add_argument("--selector-model-path", type=str, default=None)
    parser.add_argument("--selector-image-aspect-ratio", type=str, default="anyres_max_9")

    # selection
    parser.add_argument("--sample_frames", type=int, default=64)
    parser.add_argument("--input_frames", type=int, default=1)
    parser.add_argument("--selector_method", type=str, default="topk")
    parser.add_argument("--score-docs", type=str, default=None)

    # resume
    parser.add_argument("--main-process", action='store_true')
    parser.add_argument("--sub-vid-cache", type=str, default=None)

    # task
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subtitles", action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.sub_vid_cache is not None:
        os.makedirs(args.sub_vid_cache, exist_ok=True)
    args.do_sample = args.temperature > 0
    
    args.annot_scores = args.selector_method.startswith('annot_scores')

    eval_model(args)

