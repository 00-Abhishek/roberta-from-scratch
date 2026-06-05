<div align="center">

# 🤖 RoBERTa from Scratch

### Full pretraining pipeline — custom BPE tokenizer → masked language modelling → inference
### No pretrained weights. No shortcuts. Just raw architecture, math, and training loops.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Tokenizers-FFD21E?style=flat-square)](https://huggingface.co)
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Educational-blue?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/00-Abhishek/roberta-from-scratch/main.yml?style=flat-square&label=CI)](https://github.com/00-Abhishek/roberta-from-scratch/actions)

</div>

---

## 🧠 What is This?

Most tutorials tell you to `from transformers import RobertaModel` and call it a day.
This repo doesn't.

Here, **every stage of the pretraining pipeline is built and executed explicitly** — from raw corpus text to a working fill-mask model — so you understand exactly what is happening at each step. If you've ever wondered what it actually takes to pretrain a language model, this is it.

| Stage | What happens |
|---|---|
| **Corpus** | Raw text prepared from a local dataset |
| **Tokenizer** | Byte-Level BPE tokenizer trained from scratch (50k vocab) |
| **Dataset** | Text tokenized, chunked, and formatted for MLM |
| **Collation** | Dynamic 15% masking applied per batch |
| **Model Init** | RoBERTa config initialized with random weights |
| **Pretraining** | MLM objective — predict masked tokens |
| **Inference** | Fill-mask pipeline on custom trained weights |

---

## 📐 Architecture Overview

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│   Byte-Level BPE Tokenizer      │  vocab_size = 50,000
│   (trained from scratch)        │
└─────────────────────────────────┘
    │  token_ids + attention_mask
    ▼
┌─────────────────────────────────┐
│   MLM Data Collator             │  15% tokens randomly masked
│   [MASK] token injection        │  → [MASK], random, or unchanged
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   RoBERTa Encoder Stack         │
│   ├── Input Embedding           │  token + position embeddings
│   ├── 12× Encoder Layers        │  each: MHA + FFN + LayerNorm
│   │   ├── Multi-Head Attention  │  12 heads, d_k = 64
│   │   ├── Add & Norm            │  residual connection
│   │   ├── Position-wise FFN     │  d_model=768 → d_ff=3072 → 768
│   │   └── Add & Norm            │
│   └── Pooler                    │
└─────────────────────────────────┘
    │  contextual representations
    ▼
┌─────────────────────────────────┐
│   MLM Head                      │  linear → GELU → LayerNorm
│   → vocab projection            │  (B, seq, 768) → (B, seq, 50000)
└─────────────────────────────────┘
    │  cross-entropy loss on [MASK] positions only
    ▼
  Loss ↓ (15.x → 5.8)
```

---

## 📦 Pipeline & File Map

The project is organized as a **sequential pipeline**. Each file is a self-contained, runnable stage.

```
roberta-from-scratch/
│
├── 📂 Stage 1 — Corpus & Tokenizer
│   ├── prepare_corpus.py       # Clean and format raw text for training
│   ├── train_tokenizer.py      # Train Byte-Level BPE tokenizer (50k vocab)
│   ├── export_tokenizer.py     # Save tokenizer to disk
│   ├── load_tokenizer.py       # Load and verify saved tokenizer
│   └── tokenizer_check.py      # Sanity checks on tokenizer output
│
├── 📂 Stage 2 — Dataset
│   ├── tokenize_dataset.py     # Tokenize full corpus with trained tokenizer
│   ├── build_dataset.py        # Build HuggingFace Dataset from token files
│   ├── load_dataset.py         # Load dataset splits for training
│   ├── mlm_dataset.py          # Custom MLM Dataset class
│   ├── mlm_collator.py         # Dynamic masking data collator (15%)
│   └── data_collator_check.py  # Verify masking strategy on sample batch
│
├── 📂 Stage 3 — Model
│   ├── model_config.py         # Define RoBERTa config (~45M params)
│   ├── build_config.py         # Build and export model config JSON
│   └── init_model.py           # Initialize model with random weights
│
├── 📂 Stage 4 — Training
│   ├── training_args.py        # HuggingFace TrainingArguments setup
│   ├── trainer_setup.py        # Trainer init with model, data, collator
│   ├── dry_run_train.py        # Quick 10-step sanity check run
│   ├── pretrain.py             # Main pretraining script
│   └── full_pretrain.py        # Full end-to-end pipeline in one script
│
├── 📂 Stage 5 — Testing & Inference
│   ├── check_cuda.py           # Verify GPU / CUDA availability
│   ├── forward_test.py         # Single forward pass test
│   ├── test_tokenizer.py       # Unit tests for tokenizer
│   ├── test_mlm_batch.py       # Inspect masked batch samples
│   ├── mlm_inference.py        # MLM inference on custom input
│   ├── test_fill_mask.py       # Fill-mask pipeline demo
│   └── save_final_model.py     # Export trained weights + tokenizer
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/00-Abhishek/roberta-from-scratch.git
cd roberta-from-scratch
pip install -r requirements.txt
```

> A CUDA-enabled GPU is strongly recommended. Run `python check_cuda.py` to verify.

### 2. Train the Tokenizer

```bash
python train_tokenizer.py
```
Trains a Byte-Level BPE tokenizer on your corpus. Saves to `tokenizer/`.

### 3. Prepare the Dataset

```bash
python tokenize_dataset.py
```
Tokenizes raw text and structures it for masked language modelling.

### 4. Pretrain the Model

```bash
python pretrain.py
```
Initializes a RoBERTa model with random weights and runs MLM pretraining.

### 5. Test Fill-Mask Inference

```bash
python test_fill_mask.py
```

```
Input  : "Transformers are very <mask> models."
Output : "Transformers are very powerful models."  ✓
```

---

## 📊 Training Summary

| Hyperparameter | Value |
|---|---|
| Architecture | RoBERTa-style Encoder |
| Vocabulary Size | 50,000 (BPE) |
| Model Parameters | ~45M |
| Model Dimension (`d_model`) | 768 |
| Encoder Layers | 12 |
| Attention Heads | 12 |
| Feed-Forward Dim | 3,072 |
| Training Objective | Masked Language Modeling (MLM) |
| Masking Rate | 15% |
| Epochs Trained | 1 |
| Initial Loss | ~15.x |
| Final Loss | ~5.8 |
| Hardware | CUDA GPU |

> Loss drop from **~15 → ~5.8** in a single epoch confirms the model successfully learned token co-occurrence patterns from random initialization.

---

## 💾 Model Weights

GitHub's file size limits prevent storing model checkpoints here.
Pretrained weights are available on Google Drive:

📁 **[Download Weights (Google Drive)](https://drive.google.com/drive/folders/149A6nZhQGzpEWn7hIBEp3DMyvM-AhBQp)**

After downloading, place the folders in the project root:

```
roberta-from-scratch/
├── checkpoints/          ← training checkpoints
└── roberta_pretrained/   ← final exported model
```

---

## 🎓 What You'll Learn

By working through this codebase you'll understand:

- How **Byte-Pair Encoding (BPE)** builds a vocabulary from raw text
- How **dynamic masking** differs from static masking (RoBERTa vs BERT)
- How **random weight initialization** works and why it matters
- How **MLM loss** is computed only over masked positions, not the full sequence
- How HuggingFace `Trainer` abstracts the training loop — and what's underneath it
- Why **loss alone** is not enough — and how to validate with fill-mask inference

---

## 🔧 Requirements

```
torch>=2.0
transformers>=4.30
tokenizers>=0.13
datasets>=2.12
accelerate>=0.20
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🙏 Acknowledgements

- **Liu et al. (2019)** — [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692)
- **Devlin et al. (2018)** — original BERT paper that RoBERTa improves upon
- Open-source community — PyTorch, Hugging Face Transformers & Tokenizers

---

<div align="center">

**Made with ❤️ by [Abhishek Pal](https://github.com/00-Abhishek) & [Ayush Kumar Singh](https://github.com/Ayush-2703)**
Amity University Uttar Pradesh · B.Tech Artificial Intelligence · 2026
⭐ *Star this repo if it helped you understand pretraining from scratch!*

</div>
