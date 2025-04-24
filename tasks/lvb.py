from .base import VideoTask
from .utils import get_multi_choice_info, parse_multi_choice_response, mc_process_results
import logging
from collections import defaultdict


CATEGORIES = [
    'E2O',
    'E3E',
    'O2E',
    'O3O',
    'S2A',
    'S2E',
    'S2O',
    'SAA',
    'SOS',
    'SSS',
    'T2A',
    'T2E',
    'T2O',
    'T3E',
    'T3O',
    'TAA',
    'TOS'
]


class LongVideoBench(VideoTask):
    def __init__(self, dataset, split, subtitles=False, **kwargs):
        super().__init__(dataset, split, **kwargs)

        assert self.split in ['val', 'test']
            
        self.subtitles = subtitles
        if self.subtitles:
            raise NotImplementedError('subtitles not implemented yet')
            # TODO: change prompt for subtitles

    def aggregate_results(self, docs, out_root):
        if self.split == "test":
            raise NotImplementedError('test set not implemented')
            # pred = results
            # index2ans, all_choices = get_multi_choice_info(doc)
            # parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        elif self.split == "val":
            out_file = out_root + '.log'
            logging.basicConfig(filename=out_file,
                                level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            subset_to_eval_samples = defaultdict(list)
            for doc in docs:
                pred = doc["pred"][0]
                res = mc_process_results(doc, pred)
                correct = int(res["exact_match"])

                subset_to_eval_samples[doc["question_category"]].append(correct)
                subset_to_eval_samples[doc["duration_group"]].append(correct)
                subset_to_eval_samples["overall"].append(correct)
            for subset in CATEGORIES:
                sub_eval_samples = subset_to_eval_samples[subset]
                total_correct = float(sum(sub_eval_samples))
                total_answered = len(sub_eval_samples)
                logging.info(f"Evaluation on Question Categories: {subset}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
            for subset in [15, 60, 600, 3600]:
                sub_eval_samples = subset_to_eval_samples[subset]
                total_correct = float(sum(sub_eval_samples))
                total_answered = len(sub_eval_samples)
                logging.info(f"Evaluation on Duration Categories: {subset}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
            for subset in ["overall"]:
                sub_eval_samples = subset_to_eval_samples[subset]
                total_correct = float(sum(sub_eval_samples))
                total_answered = len(sub_eval_samples)
                logging.info(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
