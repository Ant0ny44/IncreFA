# IncreFA：突破生成模型溯源的静态边界

[English](README.md) | [简体中文](README_zh-CN.md)

论文 **《IncreFA: Breaking the Static Wall of Generative Model Attribution》**（CVPR 2026）的官方实现。

[[论文]](https://arxiv.org/abs/2604.17736) ·
[[数据集]](https://modelscope.cn/datasets/Ant0ny/IABench/summary)

## 最新消息

- **[2026-04]** IABench 数据集已在 ModelScope 发布。
- **[2026-03]** IncreFA 被 CVPR 2026 接收。

## 项目简介

IncreFA 面向类别增量学习场景下的生成模型溯源。本仓库提供 IABench EP1
协议的训练与评估实现，主要包括：

- CLIP 特征提取
- 模型类别与生成家族的层次监督
- Hidden Replay
- 伪未知特征插值
- 增量溯源与开放集评估

## 环境安装

建议使用 Python 3.10 或更高版本，并安装支持 CUDA 的 PyTorch。

```bash
git clone https://github.com/Ant0ny44/IncreFA.git
cd IncreFA
python -m pip install -r requirements.txt
```

首次运行时会自动下载预训练的 CLIP 权重。

## 数据集

下载 [IABench](https://modelscope.cn/datasets/Ant0ny/IABench/summary)，然后将
`configs/increfa.yml` 中的 `data.arrow_data_dir` 设置为本地 Arrow 数据集路径。
配置文件定义了增量类别序列和确定性的训练/测试划分。`nano-banana` 与
`Imagen3` 仅用于开放集测试，不参与训练。

## 训练与评估

```bash
./run.sh
```

也可以直接运行：

```bash
python train.py --config configs/increfa.yml
```

训练指标写入 `logs/increfa.jsonl`，增量结果与最新可训练状态保存在
`outputs/increfa/`。

开放集实验应同时报告与阈值无关的 AUROC、已知样本误拒率和未知样本检出率。

## 引用

```bibtex
@inproceedings{qin2026increfa,
  title={IncreFA: Breaking the Static Wall of Generative Model Attribution},
  author={Haotian Qin and Dongliang Chang and Yueying Gao and Yuexuan Tan and Lei Chen and Zhanyu Ma},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## 联系方式

如对代码或数据集有疑问，请提交 Issue，或联系 Haotian Qin：
`qinhaotian@bupt.edu.cn`。

## 开源许可

本项目采用 [MIT License](LICENSE)。
