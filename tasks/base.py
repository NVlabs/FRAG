# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FRAG/blob/main/LICENSE

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from .utils import doc_to_text_mc, mc_process_results


class VideoTask:
    def __init__(self, dataset, split, **kwargs):
        self.dataset = dataset
        self.split = split

        self.post_prompt = "\nAnswer with the option's letter from the given choices directly."
        self.prompt_assistant = None
        
        self.result_key = "pred"

    def load_visual(self, visual_path, num_frames):
        vr = VideoReader(visual_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        num_frames = min(num_frames, total_frames)
        if num_frames == 1:
            # get middle instead of first for single frame
            frame_indices = np.arange(0, total_frames, total_frames / 2).astype(int).tolist()
            frame_indices = frame_indices[1:]
        else:
            frame_indices = np.arange(0, total_frames, total_frames / num_frames).astype(int).tolist()

        frame_indices = [x for x in frame_indices if x < total_frames]

        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(frames[i]).convert('RGB') for i in range(len(frames))]

        return images, frame_indices

    def doc_to_prompt(self, doc):
        prompt_user = doc_to_text_mc(doc, {"post_prompt": self.post_prompt})
        prompt_assistant = self.prompt_assistant

        return (prompt_user, prompt_assistant)

    def doc_to_visual_name(self, doc):
        return doc["video"]

    def process_results(self, doc, results):
        return results

    def aggregate_results(self, docs, out_root):
        out_file = out_root + '.log'
        
        cnt = 0
        correct = 0
        for doc in docs:
            pred = doc["pred"][0]
            res = mc_process_results(doc, pred)
            cnt += 1
            correct += int(res["exact_match"])
        accuracy = float(correct) / cnt
        print(f"Accuracy: {accuracy}")
        with open(out_file, 'a') as file:
            file.write(f"Accuracy: {accuracy}\n")


class DocumentTask:
    def __init__(self, dataset, split, **kwargs):
        self.dataset = dataset
        self.split = split

        # self.post_prompt = "\nAnswer the question concisely based on the provided images."
        self.post_prompt = "\nAnswer the question based on the provided images. Please make your response as concise as possible."
        self.prompt_assistant = None

        self.result_key = "pred"

    def load_visual(self, visual_paths, num_frames):
        assert isinstance(visual_paths, list)
        total_frames = len(visual_paths)
        if num_frames > 0:
            num_frames = min(num_frames, total_frames)
            frame_indices = np.arange(0, total_frames, total_frames / num_frames).astype(int).tolist()
        else:
            frame_indices = list(range(len(visual_paths)))
        frame_indices = [x for x in frame_indices if x < total_frames]
        images = [Image.open(visual_paths[i]).convert("RGB") for i in frame_indices]

        return images, frame_indices

    def doc_to_prompt(self, doc):
        prompt_assistant = self.prompt_assistant

        prompt_user = doc["question"].strip() + self.post_prompt

        return (prompt_user, prompt_assistant)

    def doc_to_visual_name(self, doc):
        return doc["deck_name"]

    def process_results(self, doc, results):
        return results

    def aggregate_results(self, docs, out_root):
        pass

