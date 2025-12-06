# Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2312.16476-b31b1b.svg)](https://arxiv.org/abs/2411.16602)
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://chat2svg.github.io/)

![title](./assets/teaser.png)

## Overview

Chat2SVG is a framework for generating vector graphics using large language models and image diffusion models. The system works in multiple stages to generate, enhance, and optimize SVG from text descriptions.


## Updates
- **[2025.04.02]**: The official Anthropic APIs are now supported. You can configure this in the `.env` file. You can also adjust the `max_tokens` parameter in `utils/gpt.py` on line 127. Thanks to [@potpov](https://github.com/potpov)'s contribution.
- **[2025.03.31]**: We sincerely thank [@pq-dong](https://github.com/pq-dong) for implementing a web demo for convenient use. Visit the [repository](https://github.com/pq-dong/Chat2SVG) for more details. The web demo will be refined and updated in the future.
![web demo](./assets/web_demo.png)
- [x] SVG template generation with Large Language Models
- [x] Detail enhancement with image diffusion models
- [x] SVG shape optimization


## Setup

## 1. System Prerequisites
Ensure you are on Linux with NVIDIA Drivers installed.
If running in Docker/Ubuntu, install OpenGL libs for OpenCV:
```bash
apt-get update && apt-get install -y libgl1-mesa-glx git build-essential
````

### 2. Create Environment

We use Python 3.10.

```
conda create -n chat2svg python=3.10 -y
conda activate chat2svg
```

### 3. Install CUDA & PyTorch (The Foundation)

**Critical Step:** We must match system CUDA headers with PyTorch CUDA version.

Bash

```
# 1. Install CUDA Toolkit 11.8 (Provides nvcc and headers for diffvg)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# 2. Install PyTorch compatible with CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install General Dependencies

Use our frozen requirements file to prevent version conflicts (especially Numpy 2.0).

```
# Copy the provided requirements_frozen.txt to your root dir
pip install -r requirements_frozen.txt --no-deps
```

### 5. Install Git-based Dependencies

These packages must be installed separately to avoid dependency resolution messing up our environment.

```
# OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git --no-deps

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git
 --no-deps

# Picosvg (Standard install)
pip install picosvg
```

### 6. Compile & Install DiffVG (The Boss Fight) ⚔️

The original diffvg fails to compile on modern environments. We will patch it automatically.

#### 6.1 Clone & Init

Bash

```
cd /mnt  # Or your preferred directory
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
```

#### 6.2 Auto-Patch Source Code (Fixes Version Errors)

Run these commands to modify `diffvg.cpp` without opening an editor:

Bash

```
# Fix "CUDA versions below 12 not supported" error
sed -i '1i #define CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12' diffvg.cpp
```

#### 6.3 Compile

Set environment variables so CMake can find the Conda CUDA headers.

Bash

```
export CUDA_HOME=$CONDA_PREFIX

# CRITICAL: We explicitly add the 'cccl' path to fix "thrust/execution_policy.h not found"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include/cccl:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CPATH

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Use setup.py directly
python setup.py install
```

### 7. Verification

Run this command to check if everything is perfect.

Bash

```
python -c "import torch; import diffvg; import pydiffvg; import transformers; import numpy; print(f'DiffVG: Success | GPU: {torch.cuda.is_available()} | Numpy: {numpy.__version__} (<2.0) | Transformers: {transformers.__version__}')"
```

If you see **DiffVG: Success**, you are ready to go!

---

### Troubleshooting

- **Download Timeout**: If HuggingFace models fail to download:
    
    Bash
    
    ```
    export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
    ```
    
- OpenCV Error: ImportError: libGL.so.1:
    
```
    Run apt-get install libgl1-mesa-glx or pip install opencv-python-headless "numpy<2.0".    
```



## Pipeline 🖌

> [!TIP]
> We provide two ways to generate SVG templates:
> 1. If you want to **create high-quality SVG**, we recommend checking the output of each stage to ensure the generated SVG meet "human-preferred" criteria.
> 2. If you want to **compare the performance** of our method with your own SVG generation method, we also provide a simple way to automatically generate all outputs.

> [!CAUTION]
> Hong Kong is banned by Anthropic/OpenAI. Therefore, I use a third-party API from [WildCard](https://bewildcard.com/) to forward requests to Claude. If you are in a region where you can access Anthropic/OpenAI directly, you can modify lines 64-65 in `utils/gpt.py` to use the original Anthropic API. Additional modifications may be required. Sorry for the inconvenience.

## Step-By-Step Pipeline (For High-Quality SVG 🎨)

> We have provided some sample generation and intermediate results in the `output/example_generation` folder. You can check them to get a better understanding of the pipeline.

### Stage 1: Template Generation

First, paste your Anthropic API key into the `.env` file:
```shell
OPENAI_API_KEY=<your_key>
```

Then, run the following command to generate SVG templates:
```shell
cd 1_template_generation
bash run.sh
```
- The detailed prompts of each target object can be found in `utils/util.py → get_prompt()`.
- Output files will be saved in `output/example_generation/stage_1` folder.
- To visualize/edit the SVG results, we recommend using the [SVG](https://marketplace.visualstudio.com/items?itemName=jock.svg) and [SVG Editor](https://marketplace.visualstudio.com/items?itemName=henoc.svgeditor) plugins of VSCode.
- Since multiple SVG templates are generated, we use [ImageReward](https://github.com/THUDM/ImageReward) or [CLIP](https://github.com/openai/CLIP) to select the best one for the next stage. You can also manually select the best SVG template based on your own preference.
- Finally, there should be a `target_template.svg` (e.g., `apple_template.svg`) file in the root directory.

> [!TIP]
> Our visual rectification process can solve common issues in SVG. However, we've observed that in some cases, VLM may actually degrade the quality of the SVG during rectification. We recommend double-checking the output before and after rectification to ensure the best results.

### Stage 2: Detail Enhancement

```shell
cd 2_detail_enhancement
bash download_models.sh  # download pretrained model weights
bash run.sh              # detail enhancement
```

The above command will:
- clean SVG templates using picosvg (convert shapes to cubic Bézier curves), output `apple_clean.svg`
- generate target images using [SDXL](https://civitai.com/models/269232/aam-xl-anime-mix) and [ControlNet](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0), output `apple_target.png`
- use [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to add new shapes, output `apple_with_new_shape.svg`

> [!TIP]
> 1. Adjust the `strength` to control the strength of the SDEdit (Image to Image). We recommend `0.75` for mild enhancement and `1.0` for strong enhancement.
> 2. The default number of generated target images is `4`, and we select the **first one** as the default target image. You can check all generated images to select your preferred one.
> 3. Adjust `points_per_side` in SAM to control the granularity of the added shapes, and adjust `thresh_iou` to control the threshold that determines whether a shape is a new shape or not.
> 4. As mentioned in the paper's limitation section, SAM sometimes may not add appropriate shapes. Please check the output and modify if necessary.


### Stage 3: SVG Shape Optimization
```shell
cd 3_svg_optimization
bash download_models.sh  # download pretrained SVG VAE model
bash run.sh              # optimize SVG shapes (GPU consumption: less than 4GB)
```

> [!TIP]
> 1. We turn off `enable_path_iou_loss` by default, which can greatly improve time efficiency. To avoid path semantic meaning shifts, you can set it to `True`.
> 2. We proportionally scale up the loss weights (different from the paper) to ensure faster convergence.
> 3. Results: `apple_optim_latent.svg` and `apple_optim_point.svg`

## Automated Pipeline (For Comparison ⚖️)
Code coming soon. Alternatively, you can enter each folder and run the `run.sh` script to generate all outputs.