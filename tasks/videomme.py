from .base import VideoTask
from .utils import doc_to_text_mc, OPTIONS

import os
import re
import numpy as np
from decord import VideoReader, cpu

import logging


VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame = len(vr)
    fps = vr.get_avg_fps()
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


class VideoMME(VideoTask):
    def __init__(self, dataset, split, subtitles=False, visual_folder=None, **kwargs):
        super().__init__(dataset, split, **kwargs)

        self.post_prompt = "\nThe best answer is:"
        
        assert self.split in ['test']

        self.subtitles = subtitles
        self.visual_folder = visual_folder

    def doc_to_prompt(self, doc):
        prompt_user = doc_to_text_mc(doc, {"post_prompt": self.post_prompt})
        prompt_assistant = self.prompt_assistant

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        prompt_user = option_prompt + '\n' + prompt_user

        if self.subtitles:
            video_path = os.path.join(self.visual_folder, doc["video"])
            subtitle_folder = os.path.join(os.path.dirname(self.visual_folder), 'subtitle')
            video_name = os.path.splitext(os.path.basename(doc["video"]))[0]
            subtitle_path = os.path.join(subtitle_folder, video_name + '.srt')
            if os.path.exists(subtitle_path):
                subtitle = open(subtitle_path).readlines()

                frame_num = 32  # lmms-eval, videomme_w_subtitle.yaml
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
                subtitle = subtitle_text
            else:
                subtitle = "No subtitles available"
            subtitles_prompt = "This video's subtitles are listed below:"

            prompt_user = subtitles_prompt + '\n' + subtitle + '\n' + prompt_user

        return (prompt_user, prompt_assistant)

    def aggregate_results(self, docs, out_root):
        out_file = out_root + '.log'
        logging.basicConfig(filename=out_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        category2score = {}

        for video_type in VIDEO_TYPE:
            for category in CATEGORIES:
                for sub_category in SUB_CATEGORIES:
                    for task_category in TASK_CATEGORIES:
                        key = f"{video_type}_{category}_{sub_category}_{task_category}"
                        category2score[key] = {"correct": 0, "answered": 0}

        for doc in docs:
            video_type = doc["duration"]
            category = doc["domain"]
            sub_category = doc["sub_category"]
            task_category = doc["task_type"]
            key = f"{video_type}_{category}_{sub_category}_{task_category}"
            category2score[key]["answered"] += 1
            
            pred = doc["pred"][0]
            pred_ans = extract_characters_regex(pred)
            answer = OPTIONS[doc["answer"]]
            correct = int(pred_ans == answer)
            category2score[key]["correct"] += correct

        for video_type in VIDEO_TYPE:
            total_correct = 0
            total_answered = 0
            for k, v in category2score.items():
                if video_type in k:
                    total_correct += v["correct"]
                    total_answered += v["answered"]
            logging.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        for category in CATEGORIES:
            total_correct = 0
            total_answered = 0
            for k, v in category2score.items():
                if category in k:
                    total_correct += v["correct"]
                    total_answered += v["answered"]
            logging.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        for sub_cate in SUB_CATEGORIES:
            total_correct = 0
            total_answered = 0
            for k, v in category2score.items():
                if sub_cate in k:
                    total_correct += v["correct"]
                    total_answered += v["answered"]
            logging.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        for task_cate in TASK_CATEGORIES:
            total_correct = 0
            total_answered = 0
            for k, v in category2score.items():
                if task_cate in k:
                    total_correct += v["correct"]
                    total_answered += v["answered"]
            logging.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            total_correct += v["correct"]
            total_answered += v["answered"]
        logging.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        return 100 * total_correct / total_answered if total_answered > 0 else 0

