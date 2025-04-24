# Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import random
import string

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def load_video(visual_path, num_frames):
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


def load_images(visual_paths, num_frames):
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


################ Multi Choice ################
OPTIONS = string.ascii_uppercase


def doc_to_text_mc(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    question = [doc["question"].strip()]
    options = doc["options"]
    for i, option in enumerate(options):
        question.append(f"{OPTIONS[i]}. {option.strip()}")
    question = "\n".join(question)
    if "pre_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["pre_prompt"] != "":
        question = f"{model_specific_prompt_kwargs['pre_prompt']}{question}"
    if "post_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["post_prompt"] != "":
        question = f"{question}{model_specific_prompt_kwargs['post_prompt']}"
    return question


def mc_process_results(doc, results):
    pred = results
    index2ans, all_choices = get_multi_choice_info(doc)
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    return {
        "exact_match": parsed_pred == OPTIONS[doc["answer"]],
    }


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def get_multi_choice_info(doc):
    all_choices = []
    index2ans = {}
    options = doc["options"]
    for i, option in enumerate(options):
        index2ans[OPTIONS[i]] = option.strip()
        all_choices.append(OPTIONS[i])

    return index2ans, all_choices
################ Multi Choice ################