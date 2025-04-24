# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see https://github.com/OpenGVLab/InternVL/blob/main/LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
import math
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from .conversation import get_conv_template


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name, gpu0_load=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - (1 - gpu0_load)))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * gpu0_load)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0

    return device_map


class InternVL:
    def __init__(self, model_path, generation_args, image_aspect_ratio=12):
        model_name = os.path.basename(model_path)
        max_num = int(image_aspect_ratio)

        # reduce gpu0 load for 76B, 8 gpus
        if torch.cuda.device_count() == 8 and model_name == 'InternVL2-Llama3-76B':
            gpu0_load = 0.0
        elif torch.cuda.device_count() == 8 and model_name == 'InternVL2-40B' and max_num < 6:
            gpu0_load = 0.1
        else:
            gpu0_load = 0.5

        device_map = split_model(model_name, gpu0_load=gpu0_load)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_args = generation_args
        
        self.max_num = max_num
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

    def run_model(self, images, message, output_scores=False, post_proc_func=None):
        message, prompt_assistant = message

        pixel_values = [process_image(image, max_num=self.max_num) for image in images]
        num_patches_list = [x.size(0) for x in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0).to(torch.bfloat16).cuda()
        
        num_images = len(images)
        image_tokens = []
        for i in range(num_images):
            image_tokens.append(f"Image-{i+1}: <image>")
        image_tokens = '\n'.join(image_tokens)
        question = f"{image_tokens}\n{message}"
        
        # from InternVLChatModel.chat
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], prompt_assistant)
        query = template.get_prompt()
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        
        generation_config = self.generation_args.copy()
        generation_config['eos_token_id'] = eos_token_id
        generation_config['output_scores'] = output_scores
        generation_config['return_dict_in_generate'] = output_scores
        
        with torch.inference_mode():
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        if post_proc_func is not None:
            outputs = post_proc_func(generation_output)
        else:
            outputs = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            outputs = outputs.split(template.sep)[0].strip()
            
        return outputs

        
