# Adopted from https://github.com/rubenpt91/MP-DocVQA-Framework
# Licensed under The MIT License.
# See https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/LICENSE for details

from .base import DocumentTask

import editdistance
import json


class Evaluator:
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.total_accuracies = []
        self.total_anls = []

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0

    def get_metrics(self, gt_answers, preds, answer_types=None, update_global_metrics=True):
        answer_types = answer_types if answer_types is not None else ['string' for batch_idx in range(len(gt_answers))]
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_accuracy.append(self._calculate_accuracy(gt, pred, answer_types[batch_idx]))
            batch_anls.append(self._calculate_anls(gt, pred, answer_types[batch_idx]))

        # if accumulate_metrics:
        #     self.total_accuracies.extend(batch_accuracy)
        #     self.total_anls.extend(batch_anls)

        return {'accuracy': batch_accuracy, 'anls': batch_anls}

    def get_retrieval_metric(self, gt_answer_page, pred_answer_page):
        retrieval_precision = [1 if gt == pred else 0 for gt, pred in zip(gt_answer_page, pred_answer_page)]
        return retrieval_precision

    def update_global_metrics(self, accuracy, anls, current_epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = current_epoch
            return True

        else:
            return False

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()

    def _calculate_accuracy(self, gt, pred, answer_type):

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred, answer_type):
        if len(pred) == 0:
            return 0

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls


class MPDocVQA(DocumentTask):
    def __init__(self, dataset, split, **kwargs):
        super().__init__(dataset, split, **kwargs)

        assert self.split in ['val', 'test', 'train']

    def doc_to_visual_name(self, doc):
        return str(doc["id"])  # same doc_id could have different pages, so only same id gives same visual
    
    def aggregate_results(self, docs, out_root):
        if self.split == "val":
            evaluator = Evaluator(case_sensitive=False)
            
            total_accuracies = 0
            total_anls = 0

            for doc in docs:
                metric = evaluator.get_metrics([doc['answer']], [doc['pred'][0].strip()], doc.get('answer_type', None))
                total_accuracies += sum(metric['accuracy'])
                total_anls += sum(metric['anls'])
                
            total_accuracies = total_accuracies/len(docs)
            total_anls = total_anls/len(docs)
            
            print(f"Acc: {total_accuracies*100}\nANLS: {total_anls*100}")
            out_file = out_root + '.log'
            with open(out_file, 'a') as file:
                file.write(f"Acc: {total_accuracies*100}\nANLS: {total_anls*100}\n")
        elif self.split == "test":
            out_all = []
            for doc in docs:
                out = {
                    "questionId": int(doc['id']),
                    "answer": doc['pred'][0].strip(),
                    "answer_page": "",
                }
                out_all.append(out)
            out_file = out_root + '_sub.json'
            with open(out_file, "w") as f:
                json.dump(out_all, f)

