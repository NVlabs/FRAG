import importlib


model_map = {
    'internvl': ('.internvl.internvl', 'InternVL'),
    'llava_ov': ('.llava_ov', 'LLaVAOneVision'),
    'qwenvl': ('.qwenvl', 'QwenVL')
}


def build_model(model_name, model_path, generation_args, image_aspect_ratio=None, **kwargs):
    module_name, func_name = model_map[model_name]
    module = importlib.import_module(module_name, package=__package__)
    model_init = getattr(module, func_name)
    
    return model_init(model_path, generation_args, image_aspect_ratio=image_aspect_ratio, **kwargs)

