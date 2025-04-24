import math
import hashlib
import numpy as np
from decord import VideoReader, cpu


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def hash_string(input_string):
    hash_object = hashlib.sha1()
    hash_object.update(input_string.encode('utf-8'))
    hashed_string = hash_object.hexdigest()

    return hashed_string


def frame_to_sec(frame_indices, visual_path):
    vr = VideoReader(visual_path, ctx=cpu(0), num_threads=1)

    # Get the frame rate (FPS)
    fps = vr.get_avg_fps()
    
    return [float(x)/fps for x in frame_indices]

