# IncreFA: Breaking the Static Wall of Generative Model Attribution

[English](README.md) | [简体中文](README_zh-CN.md)

Official implementation of **"IncreFA: Breaking the Static Wall of Generative Model Attribution"** (CVPR 2026).

[[Paper]](https://arxiv.org/abs/2604.17736) ·
[[Dataset]](https://modelscope.cn/datasets/Ant0ny/IABench/summary)

## News

- **[2026-04]** The IABench dataset is available on ModelScope.
- **[2026-03]** IncreFA was accepted by CVPR 2026.

## Overview

IncreFA addresses generative-model attribution under class-incremental learning.
This repository provides the training and evaluation implementation for the
IABench EP1 protocol, including:

- CLIP feature extraction
- Hierarchical model and family supervision
- Herded latent replay
- Pseudo-unseen feature interpolation
- Incremental attribution and open-set evaluation

## Installation

Python 3.10 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/Ant0ny44/IncreFA.git
cd IncreFA
python -m pip install -r requirements.txt
```

The pretrained CLIP weights are downloaded automatically on first use.

## Dataset

Download [IABench](https://modelscope.cn/datasets/Ant0ny/IABench/summary) and
set `data.arrow_data_dir` in `configs/increfa.yml` to the local Arrow dataset.
The configuration defines the incremental class stream and deterministic
train/test split. `nano-banana` and `Imagen3` are reserved for held-out
open-set evaluation.

## Training and Evaluation

```bash
./run.sh
```

Alternatively:

```bash
python train.py --config configs/increfa.yml
```

Metrics are written to `logs/increfa.jsonl`. Incremental results and the latest
trainable state are saved under `outputs/increfa/`.

Open-set results should report threshold-independent AUROC and the
known-sample false-unseen rate together with held-out detection accuracy.

## Citation

```bibtex
@inproceedings{qin2026increfa,
  title={IncreFA: Breaking the Static Wall of Generative Model Attribution},
  author={Haotian Qin and Dongliang Chang and Yueying Gao and Yuexuan Tan and Lei Chen and Zhanyu Ma},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Contact

For questions about the code or dataset, please open an issue or contact
Haotian Qin at `qinhaotian@bupt.edu.cn`.

## License

This project is released under the terms of the [MIT License](LICENSE).
