import importlib


task_map = {
    'nextqa': ('.base', 'VideoTask'),
    'egoschema': ('.egoschema', 'EgoSchema'),
    'lvb': ('.lvb', 'LongVideoBench'),
    'videomme': ('.videomme', 'VideoMME'),
    'mlvu': ('.mlvu', 'MLVU'),
    'slidevqa': ('.slidevqa', 'SlideVQA'),
    'mmlbdoc': ('.mmlbdoc', 'MMLongBenchDoc'),
    'mpdocvqa': ('.mpdocvqa', 'MPDocVQA')
}


def build_task(dataset, split, **kwargs):
    if dataset is None:
        module_name = '.base'
        fund_name = 'VideoTask'
    else:
        module_name, func_name = task_map[dataset]
    module = importlib.import_module(module_name, package=__package__)
    task_init = getattr(module, func_name)
    
    return task_init(dataset, split, **kwargs)