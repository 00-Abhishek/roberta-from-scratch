RoBERTa From Scratch (PyTorch + Hugging Face)

This repository contains an end-to-end implementation of pretraining a RoBERTa-style Transformer model from scratch, using PyTorch and Hugging Face libraries.

The project covers the entire pipeline, starting from raw text and tokenizer training, up to successful Masked Language Model (MLM) pretraining on a local GPU.

⚠️ Model checkpoints and pretrained weights are large (~5.5 GB) and are not stored directly in this repository.
They are shared separately via Google Drive (link below).

🔍 Project Overview

This is not fine-tuning a pretrained model.
Everything is built from scratch:

Custom Byte-Level BPE tokenizer

Dataset preparation for Masked Language Modeling

Manual RoBERTa configuration

Model initialization with random weights

Pretraining using Trainer

Debugging and fixing real-world CUDA and config issues

📁 Repository Structure
roberta-from-scratch/
│
├── tokenizer/                 # Trained BPE tokenizer files
│
├── data/                      # Dataset preparation scripts
│
├── *.py                       # Training, testing, and utility scripts
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore checkpoints, venv, caches
└── README.md

Important scripts

train_tokenizer.py – Train Byte-Level BPE tokenizer

tokenize_dataset.py – Tokenize raw text for MLM

build_dataset.py / mlm_dataset.py – Dataset creation

model_config.py – RoBERTa configuration from scratch

pretrain.py / full_pretrain.py – Main training scripts

test_fill_mask.py – Inference sanity check

check_cuda.py – Verify GPU availability

📦 Model Weights & Checkpoints (Google Drive)

Due to GitHub’s file size limits, the following folders are not included in this repo:

checkpoints/

roberta_pretrained/

model .safetensors files

They are available here:

🔗 Google Drive (Private / Shared):
https://drive.google.com/drive/folders/149A6nZhQGzpEWn7hIBEp3DMyvM-AhBQp?usp=sharing

How to use the weights

Download the folders from Google Drive

Extract them into the project root:

roberta-from-scratch/
├── checkpoints/
├── roberta_pretrained/


Run inference or resume training as needed

🚀 Training Summary

Dataset size: ~36,700 samples

Epochs trained: 1

Training completed successfully on GPU

Loss decreased from ~15 → ~5.8

Stable gradients and learning rate schedule

This confirms correct learning behavior.

⚠️ Important Notes & Lessons Learned

During development, several real ML engineering issues were encountered and solved:

Tokenizer ↔ model vocab size mismatch (CUDA assert)

Position embedding overflow due to special tokens

Token type embedding issues in older Transformers versions

GitHub file size limitations for model checkpoints

Windows-specific Unicode and CUDA debugging

All fixes are reflected in the current codebase.

🧑‍🤝‍🧑 Access & Sharing

This repository is private

Collaborators can be added via GitHub settings

Model weights are shared separately via Drive

This setup mirrors industry best practices

🛠️ Requirements
pip install -r requirements.txt
