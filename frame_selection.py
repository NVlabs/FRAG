# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FRAG/blob/main/LICENSE

import torch
import torch.nn.functional as F
import numpy as np
import json
import einops

from utils import split_list

from utils import hash_string
import os
import math


OPTIONS = ["A", "B"]


class FrameSelection:
    def __init__(self, model, selector_method, output_frames, sample_frames, score_docs=None, sub_vid_cache=None):
        self.model = model
        self.output_frames = output_frames
        self.sample_frames = sample_frames
        self.selector_method = selector_method

        # TODO: score batch_size in options
        if hasattr(self.model, 'model_type') and self.model.model_type == 'clip':
            self.score_batch_size = 128
        else:
            self.score_batch_size = 1

        if score_docs is not None:
            score_docs = json.load(open(score_docs, 'r'))
            score_method = 'doc'
            self.score_docs = {}
            for doc in score_docs:
                self.score_docs[doc['id']] = doc
        else:
            score_method = 'model'

        if selector_method == 'uniform':
            assert output_frames == sample_frames
            self.proposal = None
            self.score_method = None
            self.process_score = None
        elif selector_method == 'topk_frames':
            self.proposal = 'frames'
            self.score_method = score_method
            self.process_score = 'topk_flatten'
        elif selector_method == 'topk_pairs':
            self.proposal = 'pairs'
            self.score_method = score_method
            self.process_score = 'topk_flatten'
        elif selector_method.startswith('annot_scores'):
            self.proposal = selector_method.replace('annot_scores_', '')
            self.score_method = score_method
            self.process_score = None

        self.select_id = self.model.tokenizer.convert_tokens_to_ids(OPTIONS[0])

        self.sub_vid_cache = sub_vid_cache

    def generate_proposals(self, images, frame_indices):
        if self.proposal is None:
            return [images], [frame_indices]

        proposal_func = getattr(self, 'proposal_' + self.proposal)
        images = proposal_func(images)
        frame_indices = proposal_func(frame_indices)

        return images, frame_indices

    def proposal_frames(self, lst):
        return split_list(lst, len(lst))
    
    def proposal_pairs(self, lst):
        return [[lst[i], lst[i + 1]] for i in range(len(lst) - 1)]

    def score_prompt(self, doc):

        if hasattr(self.model, 'model_type') and self.model.model_type == 'clip':
            return doc['question'].strip()

        prompt = f"Question: {doc['question'].strip()}\n"

        if self.proposal == 'frames':
            task_prompt = "Does the information within the image provide the necessary details to accurately answer the given question?\n"
        elif self.proposal == 'pairs':
            task_prompt = "Does the information within the images provide the necessary details to accurately answer the given question?\n"
        else:
            raise NotImplementedError

        post_prompt = f"{OPTIONS[0]}. yes\n{OPTIONS[1]}. no\n"
        post_prompt += "Answer with the option's letter from the given choices directly."

        return prompt + task_prompt + post_prompt

    def compute_scores_model(self, doc, images):
        prompt = self.score_prompt(doc)
        prompt = (prompt, None)

        def post_proc_func(x):
            logits = x['scores'][0]
            scores = F.softmax(logits, dim=-1)[:, self.select_id].detach()
            return scores

        scores = []
        for proposal_images in images:
            score = self.model.run_model(proposal_images, prompt, output_scores=True, post_proc_func=post_proc_func)
            scores.append(score)
        scores = torch.cat(scores, dim=0)

        return scores

    def compute_scores_model_batch(self, doc, images):
        prompt = self.score_prompt(doc)
        prompt = (prompt, None)

        def post_proc_func(x):
            logits = x['scores'][0]
            scores = F.softmax(logits, dim=-1)[:, self.select_id].detach()
            return scores

        scores = []
        iters = math.ceil(len(images) / self.score_batch_size)
        images_list = split_list(images, iters)
        for proposal_images in images_list:
            assert len(proposal_images[0]) == 1
            proposal_images = [img for imgs in proposal_images for img in imgs]
            score = self.model.run_model_batch(proposal_images, prompt, output_scores=True, post_proc_func=post_proc_func)
            scores.append(score)
        scores = torch.cat(scores, dim=0)

        return scores

    def compute_scores_model_cache(self, doc, images, frame_indices, cache_dir):
        input_string = ""
        input_string += json.dumps(doc, sort_keys=True) + '\n'
        input_string += str(self.model.model)
        input_string += f"\n{self.selector_method}_{self.output_frames}_{self.sample_frames}"
        
        cache_file = hash_string(input_string)
        cache_file = os.path.join(cache_dir, cache_file)

        prompt = self.score_prompt(doc)
        prompt = (prompt, None)

        def post_proc_func(x):
            logits = x['scores'][0]
            scores = F.softmax(logits, dim=-1)[:, self.select_id].detach()
            return scores

        # load from cache
        cache_dict = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                for line in file:
                    idx, score = line.strip().split(',')
                    cache_dict[idx] = torch.Tensor([float(score)]).to(self.model.model.device)

        scores = []

        # for proposal_images in images:
        for i, proposal_images in enumerate(images):
            idx = frame_indices[i]
            idx = ','.join(map(str, idx))
            if idx in cache_dict:
                score = cache_dict[idx]
            else:
                score = self.model.run_model(proposal_images, prompt, output_scores=True, post_proc_func=post_proc_func)
                with open(cache_file, "a") as file:
                    file.write(f"{idx},{float(score.cpu().numpy()[0])}\n")
            scores.append(score)

        assert len(scores) == len(images)
        scores = torch.cat(scores, dim=0)

        return scores

    def compute_scores_doc(self, doc, frame_indices):
        score_doc = self.score_docs[doc['id']]

        score_dict = {}
        for frames, scores in zip(score_doc['frames'], score_doc['scores']):
            score_dict[tuple(frames)] = scores

        scores = []
        for proposal_indices in frame_indices:
            score = score_dict[tuple(proposal_indices)]
            scores.append(score)
        scores = torch.Tensor(scores).to(self.model.model.device)

        return scores

    def annotate_scores(self, doc, images, frame_indices):
        images, frame_indices = self.generate_proposals(images, frame_indices)

        assert self.score_method == 'model'
        if self.sub_vid_cache is not None:
            scores = self.compute_scores_model_cache(doc, images, frame_indices, self.sub_vid_cache)
        elif self.score_batch_size > 1:
            scores = self.compute_scores_model_batch(doc, images)
        else:
            scores = self.compute_scores_model(doc, images)

        if scores is not None:
            scores = list(scores.cpu().numpy())

        return scores, frame_indices
    
    def process_flatten(self, selected, images, frame_indices):
        selected_images = []
        selected_indices = []
        for s in selected:
            selected_images.append(images[s])
            selected_indices.append(frame_indices[s])

        # flatten
        selected_images = [img for imgs in selected_images for img in imgs]
        selected_indices = [idx for indices in selected_indices for idx in indices]

        # sort by index
        argsort = np.argsort(selected_indices)
        selected_images = [[selected_images[i] for i in argsort]]
        selected_indices = [[selected_indices[i] for i in argsort]]
        
        return selected_images, selected_indices

    def select_frames(self, doc, images, frame_indices, return_scores=False):
        images, frame_indices = self.generate_proposals(images, frame_indices)

        if self.score_method is None:
            return images, frame_indices
        elif self.score_method == 'model':
            if self.score_batch_size > 1:
                scores = self.compute_scores_model_batch(doc, images)
            else:
                scores = self.compute_scores_model(doc, images)
        elif self.score_method == 'doc':
            scores = self.compute_scores_doc(doc, frame_indices)

        if self.process_score == 'topk_flatten':
            assert self.output_frames % len(images[0]) == 0
            k = self.output_frames // len(images[0])
            k = min(k, scores.shape[-1])
            _, selected = torch.topk(scores, k=k, dim=-1)
            selected = selected.cpu().numpy()
            
            selected_images, selected_indices = self.process_flatten(selected, images, frame_indices)

        if return_scores:
            scores = list(scores.cpu().numpy())
            scores = [float(x) for x in scores]
            frame_indices = [tuple(x) for x in frame_indices]
            if len(frame_indices[0]) == 1:
                frame_indices = [x[0] for x in frame_indices]
            scores_dict = dict(zip(frame_indices, scores))

            return selected_images, selected_indices, scores_dict
        else:
            return selected_images, selected_indices

