from .base import VideoTask
from .utils import OPTIONS
from collections import defaultdict
import logging


def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    else:
        return s


class MLVU(VideoTask):
    def __init__(self, dataset, split, **kwargs):
        super().__init__(dataset, split, **kwargs)
        
        self.post_prompt = "\nOnly give the best option.\nBest Option: ("

        assert self.split in ['dev_mc', 'dev_gen'], "dev only now"

    def doc_to_prompt(self, doc):
        if self.split.endswith('_mc'):
            return self.doc_to_prompt_mc(doc)
        elif self.split.endswith('_gen'):
            return self.doc_to_prompt_gen(doc)

    def doc_to_prompt_mc(self, doc):
        prompt_assistant = self.prompt_assistant

        question = f"Question: {doc['question']}\n"
        question += "Options:\n"
        for idx, c in enumerate(doc['options']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        question = question.rstrip()

        prompt_user = question + self.post_prompt

        return (prompt_user, prompt_assistant)

    def doc_to_prompt_gen(self, doc):
        pass  # TODO

    def aggregate_results(self, docs, out_root):
        if self.split in ['dev_gen', 'test_mc', 'test_gen']:
            raise NotImplementedError('')
        elif self.split == 'dev_mc':
            out_file = out_root + '.log'
            logging.basicConfig(filename=out_file,
                                level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            acc_dict = defaultdict(list)
            for doc in docs:
                pred = doc["pred"][0]
                answer = OPTIONS[doc["answer"]]
                pred_ans = extract_characters_regex(pred)
                correct = int(answer == pred_ans)
                acc_dict[doc["type"]].append(correct)

            final_res = dict()
            total=0
            idx=0
            for k, v in acc_dict.items():
                idx+=1
                final_res[k] = 100 * float(sum(v)) / len(v)
                total+=final_res[k]
            final_res['Avg'] = total /idx
            for k, v in final_res.items():
                logging.info(f"{k} Acc: {v :.2f}%")

