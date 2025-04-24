# Adopted from MMLongBench-Doc from https://github.com/mayubo2333/MMLongBench-Doc. Below is the original copyright:
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

from .base import DocumentTask
import os

import re
from math import isclose
from collections import defaultdict

from openai import OpenAI
client = OpenAI()


extract_prompt = """
Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. 
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt. 

---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\n\n1. Is the service safe?\n2. Is the service effective?\n3. Is the service caring?\n4. Is the service responsive?\n5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\n\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\n2. Regulation 12: Safe care and treatment\n3. Regulation 18: Staffing\n4. Regulation 11: Need for consent\n5. Regulation 10: Dignity and respect\n6. Regulation 9: Person-centred care\n7. Regulation 17: Good governance\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\n\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---
"""

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls<=threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
    if answer_type=="Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ""
        score = (gt==pred)
    elif answer_type=="Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt==pred)
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            gt = eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            pred = eval(pred)
        if not isinstance(pred, list):
            pred = [pred]
        print(len(gt), len(pred))
        if len(gt)!=len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            print(gt, pred)
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = ("-".join(gt)=="-".join(pred))
            else:
                score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])

    return float(score)


def eval_acc_and_f1(samples):
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    
    acc = sum([sample["score"] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["answer"]!="Not answerable"])
        precision = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["pred"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0
    
    return acc, f1


def extract_answer(question, output, prompt, model_name="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
                }
            ],
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = response.choices[0].message.content
    except:
        response = "Failed"
    
    return response


class MMLongBenchDoc(DocumentTask):
    def __init__(self, dataset, split, **kwargs):
        super().__init__(dataset, split, **kwargs)
        
        assert self.split in ['old', 'test']
        self.post_prompt = ""
        self.thres = 0.4
        self.model_name = "gpt-4o"
        
    def doc_to_visual_name(self, doc):
        return doc["doc_id"]
    
    def aggregate_results(self, docs, out_root):
        os.makedirs(out_root + '_' + self.model_name, exist_ok=True)

        id_to_scores = {}
        score_path = os.getenv("SCORE_PATH")
        if score_path is not None:
            score_docs = json.load(open(score_path))
            for doc in score_docs:
                id_to_scores[doc["id"]] = doc["scores"]
        
        for doc in docs:
            if doc["id"] in id_to_scores:
                scores = id_to_scores[doc["id"]]
                frames = doc["frames"][0]
                selected_scores = [scores[x] for x in frames]
            else:
                selected_scores = [1.0]

            if max(selected_scores) > self.thres:
                response = doc['pred'][0].strip()
                extracted_res_file = os.path.join(out_root + '_' + self.model_name, doc['id'] + '.txt')
                if os.path.exists(extracted_res_file):
                    with open(extracted_res_file, 'r') as file:
                        extracted_res = file.read()
                else:
                    extracted_res = extract_answer(doc["question"], response, extract_prompt, model_name=self.model_name)
                    with open(extracted_res_file, 'w') as file:
                        file.write(extracted_res)
                try:
                    pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                except:
                    pred_ans = "Failed to extract"
            else:
                pred_ans = "Not answerable"
                doc["pred"] = pred_ans
            
            try:
                score = eval_score(doc['answer'], pred_ans, doc["answer_format"])
            except:
                score = 0.0
            assert "score" not in doc
            doc["score"] = score

        acc, f1 = eval_acc_and_f1(docs)
        print(f"Acc: {acc*100}\nF1: {f1*100}")
        out_file = out_root + '.log'
        with open(out_file, 'a') as file:
            file.write(f"{self.model_name}\nAcc: {acc*100}\nF1: {f1*100}\n")
