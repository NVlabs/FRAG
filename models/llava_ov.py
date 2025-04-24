# Adopted from LLaVA-OneVision from https://github.com/LLaVA-VL/LLaVA-NeXT. Below is the original copyright:
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

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import copy
import math


def split_model(num_layers, gpu0_load=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()

    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - (1 - gpu0_load)))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * gpu0_load)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = 0
    device_map['model.image_newline'] = 0
    device_map['model.vision_tower'] = 0
    device_map['model.vision_resampler'] = 0
    device_map['model.mm_projector'] = 0
    device_map['lm_head'] = 0
    device_map[f'model.layers.{num_layers - 1}'] = 0

    return device_map


class LLaVAOneVision:
    def __init__(self, model_path, generation_args, image_aspect_ratio="anyres_max_9"):
        model_name = "llava_qwen"
        llava_model_args = {
                "multimodal": True,
            }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = image_aspect_ratio
        llava_model_args["overwrite_config"] = overwrite_config
        
        self.device = 'cuda'
        self.conv_template = "qwen_1_5"
        
        if "llava-onevision-qwen2-72b" in model_path:
            world_size = torch.cuda.device_count()
            if world_size < 4:
                gpu0_load = 0.5
            else:
                gpu0_load = 0.2

            device_map = split_model(80, gpu0_load=gpu0_load)
        else:
            device_map = 'auto'
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, **llava_model_args)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.generation_args = generation_args
        
    def run_model(self, images, message, output_scores=False, post_proc_func=None):
        message, prompt_assistant = message

        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
        
        image_sizes = [image.size for image in images]
        
        num_images = len(images)
        image_tokens = [DEFAULT_IMAGE_TOKEN] * num_images
        image_tokens = '\n'.join(image_tokens)
        
        question = f"{image_tokens}\n{message}"
        
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], prompt_assistant)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        # Generate response
        with torch.inference_mode():
            generation_output = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                output_scores=output_scores,
                return_dict_in_generate=output_scores,
                **self.generation_args,
            )

        if post_proc_func is not None:
            outputs = post_proc_func(generation_output)
        else:           
            outputs = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        return outputs
