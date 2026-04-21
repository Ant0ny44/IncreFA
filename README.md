# IncreFA: Breaking the Static Wall of Generative Model Attribution

Official implementation of the paper **"IncreFA: Breaking the Static Wall of Generative Model Attribution"** (CVPR 2026).

[[Paper]](https://arxiv.org/abs/2604.17736) [[Dataset]](https://modelscope.cn/datasets/Ant0ny/IABench/summary)

-----

## 📢 News

  * **[2026-04]** IABench dataset (512GB) is now available on ModelScope\!
  * **[2026-03]** IncreFA has been accepted by CVPR 2026.

-----

## 🛠️ Code Maintenance

> [\!IMPORTANT]  
> We are currently **cleaning up and refactoring** the training and evaluation code to ensure reproducibility. The full source code, including pre-trained weights and configuration files, will be released sequentially. Please **Stay Tuned\!** 🌟

-----

## 📊 IABench Dataset

**IABench** (Incremental Attribution Benchmark) is a large-scale dataset specifically designed for generative model attribution in incremental learning scenarios.

  * **Scale:** \~570,000 images, 512GB.
  * **Coverage:** 28 generative models (2022–2025).
  * **Format:** Optimized **Arrow binary format** for high-performance I/O and seamless streaming.

### 📥 Data Access

The dataset is hosted on ModelScope. You can load it directly using the `modelscope` library without manual decompression.

#### 1\. Installation

```bash
pip install modelscope datasets
```

## ✒️ Citation

If you find our work or the IABench dataset useful for your research, please cite:

```bibtex
@inproceedings{qin2026increfa,
  title={IncreFA: Breaking the Static Wall of Generative Model Attribution},
  author={Haotian Qin, Dongliang Chang, Yueying Gao, Lei Chen, and Zhanyu Ma},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

-----

## 📩 Contact

For any questions regarding the code or dataset, please open an issue or contact **Haotian Qin** at `qinhaotian@bupt.edu.cn`.
