# Adopted from Qwen2-VL from https://github.com/QwenLM/Qwen2-VL. Below is the original copyright:
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

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class QwenVL:
    def __init__(self, model_path, generation_args, image_aspect_ratio=None):        
        # load model and processor
        self.model= Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # set dynamic resolution param
        if image_aspect_ratio is not None:
            self.max_pixels = int(image_aspect_ratio) * 28 * 28
        else:
            self.max_pixels = None
        
        # get tokenizer from processor. needed to get option's id in frame selector
        self.tokenizer = self.processor.tokenizer
        
        # save generation_args to use in run_model
        self.generation_args = generation_args
        
    def run_model(self, images, message, output_scores=False, post_proc_func=None):
        message, prompt_assistant = message
        
        messages = []
        
        user_content = []
        for image in images:
            if self.max_pixels is not None:
                user_content.append({"type": "image", "image": image, "max_pixels": self.max_pixels})
            else:
                user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": message})
        user_message = {
            "role": "user",
            "content": user_content
        }
        messages.append(user_message)
        
        if prompt_assistant is not None:
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": prompt_assistant}]            
            }
            messages.append(assistant_message)
            
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference
        with torch.inference_mode():
            generation_output = self.model.generate(
                **inputs,
                **self.generation_args,
                output_scores=output_scores,
                return_dict_in_generate=output_scores,
            )
            
        if post_proc_func is not None:
            outputs = post_proc_func(generation_output)
        else:           
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generation_output)
            ]
            outputs = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs = outputs[0].strip()
        return outputs     

        
            
        