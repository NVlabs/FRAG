from .base import DocumentTask
import string
from collections import Counter
import re


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

def normalize_answer(s, question):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    def yesno(text):
        if 'yes' == text[:3] or 'no' == text[:2]:
            text = text.split()[0]
        return text
    def replace_text(text):
        return text.replace('this is ', '').replace('it is ', '').replace('&', ',').replace('and', ',').replace('percent', '').replace('organisation', 'organization').replace('because of', '').replace('because', '').replace('due to', '').replace('hours', 'hrs').replace('minites', 'min')
    def word2number(text):
        words = text.split()
        return ' '.join([str(WORD_NUMBER_MAP[word]) if word in WORD_NUMBER_MAP else word for word in words])
    def remove_unit(text, question):
        if 'how many' in question:
            idx = question.find('how many')
            unit = question[idx+len('how many'):].split()[0]
            text = text.replace(unit, '')
        if 'which' in question:
            idx = question.find('which')
            unit = question[idx+len('which'):].split()[0]
            text = text.replace(unit, '')
        return text
    return word2number(white_space_fix(yesno(remove_articles(remove_punc(remove_unit(replace_text(lower(s)), question))))))


class SlideVQA(DocumentTask):
    def __init__(self, dataset, split, **kwargs):
        super().__init__(dataset, split, **kwargs)

        assert self.split in ['dev', 'test', 'train']

    def aggregate_results(self, docs, out_root):
        f1 = exact_match = 0
        precisions = {}
        recalls = {}
        ems = {}
        for doc in docs:
            qa_id = doc['id']
            question = doc['question']
            prediction = doc['pred'][0].strip()
            ground_truth = doc['answer']
            prediction_tokens = normalize_answer(prediction, question).split()
            ground_truth_tokens = normalize_answer(ground_truth, question).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                precisions[qa_id] = recalls[qa_id] = ems[qa_id] = 0
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 += (2 * precision * recall) / (precision + recall)
            exact_match += (prediction_tokens == ground_truth_tokens)
            precisions[qa_id] = precision
            recalls[qa_id] = recall
            ems[qa_id] = (prediction_tokens == ground_truth_tokens)
        exact_match = exact_match / len(docs)
        f1 = f1 / len(docs)

        out_file = out_root + '.log'
        with open(out_file, 'a') as file:
            file.write(f"EM: {exact_match*100}\nF1: {f1*100}")

