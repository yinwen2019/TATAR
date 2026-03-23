# 📸 One Model, Two Minds  
## Task-Conditioned Reasoning for Unified Image Quality and Aesthetic Assessment

[![arXiv](https://img.shields.io/badge/arXiv-2603.19779-b31b1b)](https://arxiv.org/abs/2603.19779)

Official implementation of our unified framework for:

- **Image Quality Assessment (IQA)**
- **Image Aesthetic Assessment (IAA)**

---

## 🔍 Motivation

Although IQA and IAA are both image scoring tasks, they fundamentally differ in cognitive demand and optimization objectives.

### Problem 1: Task-Conditioned Reasoning Necessity

- **IQA** is a *low perceptual perplexity* task.  
  It focuses on objective visual degradations such as blur, noise, and compression artifacts.  
  Fast perceptual judgment (System-1 style reasoning) is sufficient.

- **IAA** is a *high reasoning complexity* task.  
  It involves composition, semantics, emotional tone, and stylistic preference.  
  Deliberative reasoning (System-2 style analysis) is often required to distinguish subtle differences.

A unified model that always reasons — or never reasons — inevitably leads to task mismatch.

---

### Problem 2: Task-Dependent Reward Complexity

- **IQA** aligns with an *absolute regression objective*:  
  the predicted score approximates a perceptual ground-truth MOS.

- **IAA** aligns with a *relative preference objective*:  
  aesthetic judgment is inherently comparative and better modeled via ranking or pairwise alignment.

Using identical supervision for both tasks causes optimization conflict under shared parameters.

---

## 🧠 Our Approach

We propose a **two-stage training framework** that enables task-conditioned reasoning under score-only supervision.

---

## 🏗 Training Pipeline

### Stage 1 — Asymmetric Supervised Fine-Tuning (SFT)

Location:
```
open-r1-multimodal/src/sft/
```

Goal:
- Break the "always-think" inductive bias.
- Apply asymmetric thought dropout:
  - IQA → high dropout of thinking tokens
  - IAA → retain thinking tokens

Supervision:
- Score-only (MOS)
- No ground-truth chain-of-thought required

This stage initializes task-aware reasoning priors.

---

### Stage 2 — Task-Conditioned Reinforcement Learning (GRPO)

Location:
```
open-r1-multimodal/src/openr1/
```

Goal:
- Learn when reasoning is necessary under task condition.

Reward Design:

**IQA (Low Perplexity)**
- Absolute score regression reward
- Strong penalty for redundant reasoning

**IAA (High Perplexity)**
- Pairwise / ranking preference reward
- Encourages discriminative reasoning

Thinking tokens incur a cost penalty to balance efficiency and performance.

This stage induces task-dependent reasoning behavior without modifying model architecture.

---

## 📂 Repository Structure

```
.
├── benchmark/VR/               # Ground-truth benchmark files
│
├── src/
│   ├── eval/                   # Evaluation entry scripts
│   │   ├── results/
│   │   ├── eval_uni_iqa_iaa.py
│   │   ├── eval_vqa.py
│   │   ├── eval_vr.py
│   │   └── *.sh
│   │
│   └── internvl/               # Backbone utilities
│
├── open-r1-multimodal/         # Main training project
│   ├── configs/
│   ├── data/
│   ├── data_config/
│   ├── local_scripts/
│   ├── src/
│   │   ├── openr1/             # Stage 2 RL training
│   │   └── sft/                # Stage 1 SFT training
│   ├── wandb/
│   ├── run_iqa_iaa.sh
│   ├── run_qinsight_*.sh
│   ├── setup.cfg
│   └── setup.py
```

---

## 📊 Evaluation

Evaluation scripts are located in:

```
src/eval/
```

Includes:
- Unified IQA + IAA evaluation
- VQA evaluation
- VR evaluation

Ground-truth files are stored under:

```
benchmark/VR/
```

---

## 📈 Supported Datasets

### IQA
- KonIQ
- SPAQ
- KADID
- Other distortion-based datasets

### IAA
- AVA
- PAPA
- TAD66K
- Flickr-AES

---

## 🚀 Quick Start

**Coming Soon**

We will release:
- Environment setup instructions
- Dataset preprocessing scripts
- Training commands
- Pretrained checkpoints
- Inference demo

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{yourpaper2026,
  title={One Model, Two Minds: Task-Conditioned Reasoning for Unified Image Quality and Aesthetic Assessment},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:2603.19779},
  year={2026}
}
```

---

## 📬 Contact

For questions or issues, please open an issue in this repository.

---

## 📝 License

To be released.