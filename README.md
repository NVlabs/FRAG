# FRAG: Frame Selection Augmented Generation

[De-An Huang](https://ai.stanford.edu/~dahuang/), [Subhashree Radhakrishnan](), [Zhiding Yu](https://chrisding.github.io/), [Jan Kautz](https://jankautz.com/)

[[`arXiv`](https://arxiv.org/abs/2504.17447)] [[`Project`]()] [[`BibTeX`](#Citation)]


## Contents
- [Install](#install)
- [Inference](#inference)
- [Evaluation](#evaluation)


## Install

The core of FRAG is zero-shot and has minimal dependencies. Follow the instructions below to install the base models and benchmarks.

### Data Loading
```Shell
pip install decord
```

### Models

1. **LLaVA-OneVision**: Follow the instructions [here](https://github.com/LLaVA-VL/LLaVA-NeXT) to install LLaVA-OneVision. Please make sure that you can run the examples [here](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov#generation).

2. **InternVL-2**: If you have already installed LLaVA-OneVision, the dependencies should already work for InternVL-2. If you only want to use InternVL-2, follow the instructions [here](https://huggingface.co/OpenGVLab/InternVL2-8B#quick-start). Please make sure that you can run the examples [here](https://huggingface.co/OpenGVLab/InternVL2-8B#inference-with-transformers).

3. **Qwen2-VL**: Follow the instructions [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#quickstart). Please make sure that you can run the multi image inference example. We recommend using `transformers==4.45.0` for Qwen2-VL.

Note that you only need to install the models you would like to use. If you want to quickly try out FRAG, we recommend starting with InternVL-2 first, as it has fewer dependencies.


### Benchmarks

1. **MP-DocVQA**:
```Shell
pip install editdistance
```

2. **MMLongBench-Doc**:
```Shell
pip install openai
```

## Inference

### Video

Video inference example:
```Shell
python run_model.py \
    --output-dir . \
    --model "internvl" \
    --model-path "OpenGVLab/InternVL2-8B" \
    --image-aspect-ratio "1" \
    --selector-model "internvl" \
    --selector-model-path "OpenGVLab/InternVL2-8B" \
    --selector-image-aspect-ratio "12" \
    --sample_frames 64 \
    --input_frames 8 \
    --selector_method "topk_frames" \
    --input-type "video" \
    --input-path <path-to-video> \
    --query "Are there any fish in the video?"
```
This uses InternVL2-8B for both answering and scoring. Scoring uses a maximum of 12 tiles for dynamic resolution, while answering disables dynamic resolution (`--image-aspect-ratio "1"`). The example first uniformly samples 64 frames, and selects the top 8 frames for answering (FRAG-64-Top8). We use `--sample_frames 256` in the paper. For longer or shorter videos, around 1 fps for `--sample_frames` is a good starting point. The example also generates `<video-path>.json`, which contains the inputs and outputs of the model.


### Document

Document inference example:
```Shell
python run_model.py \
    --output-dir . \
    --model "internvl" \
    --model-path "OpenGVLab/InternVL2-8B" \
    --image-aspect-ratio "16" \
    --selector-model "internvl" \
    --selector-model-path "OpenGVLab/InternVL2-8B" \
    --selector-image-aspect-ratio "12" \
    --sample_frames -1 \
    --input_frames 1 \
    --selector_method "topk_frames" \
    --input-type "images" \
    --input-path <path-to-image-folder> \
    --query "What is the title of the paper?"
```
The main difference from the video example is `--input-type "images"`, which suggests that `--input-path` points to a folder containing images (from pages of a document). Our data loading function assumes that the image file names are sorted by the page order. Other differences include: `--image-aspect-ratio "16"` to use higher resolution for answering, `--sample_frames -1` to sample all the pages, and `--input_frames 1` to only select the Top-1 page for answering.


## Evaluation

We provide example scripts for benchmark evaluation using InternVL2-8B.

### Video

0. Update Paths

Update the dataset and output paths in `scripts/video/path.sh`. JSON files pointed by `$doc_path` can be downloaded [here](https://huggingface.co/datasets/deahuang/FRAG-Datasets). Follow the official download instruction for each dataset, and `$visual_path` would point to the root folder for videos.


1. Precompute FRAG Scores

```Shell
bash scripts/video/annot_scores_internvl-8b_frames.sh $dataset $num_frames $CHUNKS $IDX
```
`$dataset` is the dataset name to evaluate. `$num_frames` is the number of frames to uniformly sample from the video for FRAG scoring. `$CHUNKS` and `$IDX` would split the samples in the dataset in to `$CHUNKS` splits and only compute scores for the `$IDX` split. For example, to evaluate LongVideoBench with 256 sampled frames (as in the paper) with a single job:
```Shell
bash scripts/video/annot_scores_internvl-8b_frames.sh lvb 256 1 0
```
Here, there is only 1 chunk, and the only `$IDX` is 0. Set `$CHUNKS` to `N` and  `$IDX` in `[0, N)` to run `N` jobs for score computation. 

2. Collect FRAG Scores

The previous step computes FRAG scores for videos in the dataset, which are saved in separate files for easier parallelization. Now we collect all the FRAG scores into a single JSON file. Following the previous example, collect the FRAG scores using:
```Shell
python collect_results.py \
    --doc-path $root/datasets/LongVideoBench/lvb_val_doc_list.json \
    --result-path $output_root/lvb/val/annot_scores_internvl-8b_frames_256
```
`$root` and `$output_root` are specified in `scripts/video/path.sh` in step 0. This should generates `$output_root/lvb/val/annot_scores_internvl-8b_frames_256.json`, which will be used in the next step.

3. Evaluate FRAG

```Shell
bash scripts/video/eval_internvl-8b-max1_top32_frames.sh $dataset $num_frames
```
This script evaluates FRAG-Top32-N, where N is `$num_frames`. For LongVideoBench and 256 sampled frames:
```Shell
bash scripts/video/eval_internvl-8b-max1_top32_frames.sh lvb 256
```


### Document

We use SlideVQA and InternVL2-8B as an example. The scripts are similar to the ones for videos.


0. Update Paths

Update the dataset and output paths in `scripts/document/path.sh`.

1. Precompute FRAG Scores

```Shell
bash scripts/document/annot_scores_internvl-8b_frames.sh slidevqa -1 1 0
```
The arguments are the same as video's step 1. Here, -1 means that all the pages are sampled, and the pages are not uniformly sampled.

2. Collect FRAG Scores

```Shell
python collect_results.py \
    --doc-path ${root}/datasets/SlideVQA/test_doc.json \
    --result-path $output_root/slidevqa/test/annot_scores_internvl-8b_frames_-1
```
This should generates `$output_root/slidevqa/test/annot_scores_internvl-8b_frames_-1.json`, which will be used in the next step.

3. Evaluate FRAG

```Shell
bash scripts/document/eval_internvl-8b-max16_topk_frames.sh $dataset $num_frames
```
This script evaluates FRAG by selecting the top `$num_frames` frames. Here, `$num_frames` is K instead of N for FRAG-TopK-N because for documents we go through all the pages. For SlideVQA and Top 2 frames:
```Shell
bash scripts/document/eval_internvl-8b-max16_topk_frames.sh slidevqa 2
```

## License

Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).


## <a name="Citation"></a> Citation

If you find FRAG useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{huang2025frag,
  title={FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding},
  author={De-An Huang and Subhashree Radhakrishnan and Zhiding Yu and Jan Kautz},
  journal={arXiv preprint arXiv:2504.17447},
  year={2025}
}
```






