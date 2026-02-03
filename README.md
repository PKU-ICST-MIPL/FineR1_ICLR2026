<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning</h1>
  <p align="center">
    <a href="http://39.108.48.32/mipl/news/news.php?id=EGhehulingxiao"><strong>Hulingxiao He</strong></a>
    ·
    <a href="http://39.108.48.32/mipl/news/news.php?id=EGgengzijun"><strong>Zijun Geng</strong></a>
    ·
    <a href="http://39.108.48.32/mipl/yuxinpeng/"><strong>Yuxin Peng</strong></a>
  </p>
  <h2 align="center">ICLR 2026</h2>
<div align="center"></div>


<!-- [![arXiv](https://img.shields.io/badge/arXiv-2507.06448-b31b1b.svg)](https://arxiv.org/abs/2507.06448) -->
<!-- [![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/mikewangwzhl/PAPO) -->
<!-- [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/StevenHH2000/papo-qwen-686d92dd3d43b1ce698f851a)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Data-yellow)](https://huggingface.co/collections/PAPOGalaxy/data-686da53d67664506f652774f) -->

</div>

## 🔥 News 
- **Feb 2026:** Code, data, and model will be released soon. Please stay tuned!
- **Jan 2026:** Fine-R1 is accepted to ICLR 2026.

## 🌟 **Key Highlights**

- **To the best of our knowledge, our Fine-R1 is the first MLLM to surpass various strong CLIP-like models (e.g., SigLIP-L) in FGVR**: It's widely acknowledged that general MLLMs underperform contrastive models like CLIP/SigLIP on fine-grained tasks. Our work bridges this gap and strongly indicates the potential of generative MLLMs for discriminative vision tasks.

<div align="center">
<img src="./static/images/teaser.png" alt="Overview" width="800"/>
</div>

## 📖 **Methodology**

**Fine-R1** generates Chain-of-Thought (CoT) before producing the final fine-grained visual
recognition (FGVR) answer. It utilizes **CoT supervised fine-tuning (SFT)** and **Triplet Augmented
Policy Optimization (TAPO)**, learning the reasoning process with only few-shot samples per category.

<div align="center">
<img src="./static/images/pipeline.png" alt="Method" width="940"/>
</div>

### **Main Results**

(1) **Closed-world evaluation**: In comparison to general and reasoning MLLMs, and contrastive CLIP models, Fine-R1 excels in identifying both seen and unseen categories:

<div align="center">
<img src="./static/images/main_results_closed.png" alt="Main Results" width="1200"/>
</div>

(2) **Open-world evaluation**: Fine-R1 establishes new state-of-the-art performance with only 4-shot training samples per sub-category, achieving 74.80% relative semantic similarity on average:

<div align="center">
<img src="./static/images/main_results_open.png" alt="Main Results" width="1200"/>
</div>


## 🥰 Acknowledgements

We thank the [PAPO](https://github.com/MikeWangWZHL/PAPO/tree/main), [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout/tree/09347ddd88135b83a336a204ecf6353121bbee79), [LlamaFactory](https://github.com/hiyouga/LlamaFactory), and [EasyR1](https://github.com/hiyouga/EasyR1) team for providing the foundational codebase that we adapted to implement Fine-R1. 

## 📝 Citation

```bibtex
@inproceedings{
anonymous2026finer,
title={Fine-R1: Make Multi-modal {LLM}s Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning},
author={Anonymous},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=kyzHM557gE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">



</div>