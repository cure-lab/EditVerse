# EditVerse


This repository provides the **instruction-based video editing evaluation code** for EditVerseBench, introduced in the paper *"EditVerse: A Unified Framework for Editing and Generation via In-Context Learning"*.


> [Xuan Ju](https://juxuan27.github.io/)<sup>12</sup>, [Tianyu Wang](https://scholar.google.com/citations?user=yRwZIN8AAAAJ&hl=zh-CN)<sup>1</sup>, [Yuqian Zhou](https://yzhouas.github.io/)<sup>1</sup>, [He Zhang](https://sites.google.com/site/hezhangsprinter)<sup>1</sup>, [Qing Liu](https://qliu24.github.io/)<sup>1</sup>, [Nanxuan Zhao](https://www.nxzhao.com/)<sup>1</sup>, [Zhifei Zhang](https://zzutk.github.io/)<sup>1</sup>, [Yijun Li](https://yijunmaverick.github.io/)<sup>1</sup>, [Yuanhao Cai](https://caiyuanhao1998.github.io/)<sup>3</sup>, [Shaoteng Liu](https://www.shaotengliu.com/)<sup>1</sup>, [Daniil Pakhomov](https://scholar.google.com/citations?user=UI10l34AAAAJ&hl=en)<sup>1</sup>, [Zhe Lin](https://sites.google.com/site/zhelin625/)<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>1*</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2*</sup><br>
> <sup>1</sup>Adobe Research <sup>2</sup>The Chinese University of Hong Kong <sup>3</sup>Johns Hopkins University <sup>*</sup>Corresponding Author


<p align="center">
  <a href="http://editverse.s3-website-us-east-1.amazonaws.com/">🌐 Project Page</a> |
  <a href="https://arxiv.org/abs/2509.20360">📜 Arxiv</a> |
  <a href="https://huggingface.co/datasets/sooyek/EditVerseBench">🤗 Benchmark</a> |
  <a href="https://docs.google.com/presentation/d/1dBg3lZDFa8mRRIrOVEU_xDgzedufbwzr/edit?usp=sharing&ouid=100286465794673637256&rtpof=true&sd=true">📹 Slides</a> |
  <a href="http://editverse.s3-website-us-east-1.amazonaws.com/comparison.html">👀 Comparison</a>
</p>



## 🚀 Setup Environment

**(Optional) Create a Conda environment**

```
conda create -n EditVerse python=3.10
conda activate EditVerse
```

**Install Pytorch** 

(Adjust version/CUDA support for your system.)

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**Install dependencies**

```
pip install -r requirements.txt
```


## Download Benchmark & Generated Results

**Download benchmark**

```
git lfs install
git clone https://huggingface.co/datasets/juxuan27/EditVerseBench
```

**Unpack generated results**

```
cd EditVerseBench
tar -zxvf EditVerse_Comparison_Results.tar.gz
rm EditVerse_Comparison_Results.tar.gz
```


## Evaluation

**Command**

```
python eval.py --metrics [metrics] \
  --test_json_path EditVerseBench/EditVerseBench/EditVerseBench.json \
  --generate_results_dir [results_dir] \
  --output_csv [output_csv] \
  --gpt_api_key [your_api_key]
```

**Arguments**

- `metrics`: 
    - Use `all` to evaluate all metrics.
    - Or provide a comma-separated list (e.g., `clip_temporal_consistency`,`dino_temporal_consistency`).
    - Supported metrics include:
        - clip_temporal_consistency
        - dino_temporal_consistency
        - frame_text_alignment
        - video_text_alignment
        - pick_score_video_quality
        - editing_vlm_evaluation
- `test_json_path`: Path to the benchmark JSON file.
- `generate_results_dir`: Directory with generated results (must follow required structure).
- `output_csv`: Path to save evaluation results.
- `gpt_api_key`: OpenAI API key (needed for `editing_vlm_evaluation`).


**Example**

To evaluate the provided EditVerse results and save output to EditVerse_eval.csv:

```
python eval.py --metrics all \
--test_json_path EditVerseBench/EditVerseBench/EditVerseBench.json \
--generate_results_dir EditVerseBench/EditVerse_Comparison_Results/EditVerse \
--output_csv EditVerse_eval.csv \
--gpt_api_key [Your API key]
```

👉 Pre-computed results are available here: [EditVerseBench/automatic_evaluation_results](https://huggingface.co/datasets/juxuan27/EditVerseBench/tree/main/automatic_evaluation_results).



## Evaluate Your Own Model

**1: Benchmark JSON format**

See [EditVerseBench/EditVerseBench/test.json](https://huggingface.co/datasets/sooyek/EditVerseBench/blob/main/EditVerseBench/EditVerseBench.json) for reference.

Example entry:

```
{
    "0": {
        "<text>": "<video1> Add a small golden crown ...",
        "<video1>": "videos/174008-850361316.mp4",
        "<video1> link": "https://pixabay.com/videos/woman-smile-communication-gesture-174008/",
        "direction": "horizontal",
        "target_prompt": "A young woman stands outside in front of ...",
        "type": "add object",
        "source_prompt": "A young woman stands outside in front of ..."
    },
    "1": {
        ...
    },
    ...
}
```
Key fields:
- `<text>`: A natural language instruction describing the required edit in an interleaved format.
  - The instruction may include special tags such as `<video1>`, `<video2>`, or `<image1>`.
  - Each tag corresponds to a specific key field defined in the same JSON entry.
- `<video1>`: Local source video path.
- `<video1> link`: Source video URL.
- `direction`: "horizontal" or "vertical".
- `target_prompt`: Expected edited video description.
- `type`: Edit category.
- `source_prompt`: Text description of the original video.

**Step 2: Arrange results**

After generating results with your model, arrange files as follows:

```
Your_Folder/
  ├── 0/
  │   ├── generate.mp4   # your model output
  │   └── video1.mp4     # source video
  ├── 1/
  │   ├── generate.mp4
  │   └── video1.mp4
  ...
```

**Step 3: Run evaluation**

```
python eval.py --metrics all \
--test_json_path EditVerseBench/EditVerseBench/test.json \
--generate_results_dir [Your_Folder] \
--output_csv [Your_Results.csv] \
--gpt_api_key [your_api_key]

```

## Acknowledgement & Notice

Our code is modified on the basis of [V2VBench](https://github.com/wenhao728/awesome-diffusion-v2v), thanks to all the contributors!

This code was developed independently by Xuan Ju after leaving Adobe Inc. It is not associated with, nor does it reflect, any internal code or proprietary resources of Adobe.