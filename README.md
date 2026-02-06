# roberta-from-scratch

roberta-from-scratch is a Python project for **pretraining a RoBERTa-style Transformer model from scratch** using PyTorch and Hugging Face.

This project covers tokenizer training, dataset preparation, model initialization with random weights, and Masked Language Model (MLM) pretraining on a local GPU.

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/roberta-from-scratch.git
cd roberta-from-scratch
pip install -r requirements.txt

A CUDA-enabled GPU is recommended for training.

Usage
1. Train the tokenizer
python train_tokenizer.py


This trains a Byte-Level BPE tokenizer and saves it to the tokenizer/ directory.

2. Prepare the dataset
python tokenize_dataset.py


This tokenizes raw text and prepares the dataset for Masked Language Modeling (MLM).

3. Pretrain the model
python pretrain.py


This initializes a RoBERTa model with random weights and starts MLM pretraining.

4. Test the model (fill-mask)
python test_fill_mask.py


Example:

Transformers are very <mask> models.

Model Weights

Due to GitHub file size limits, model checkpoints and pretrained weights are not included in this repository.

They are available via Google Drive:

https://drive.google.com/drive/folders/149A6nZhQGzpEWn7hIBEp3DMyvM-AhBQp


After downloading, extract the folders into the project root:

roberta-from-scratch/
├── checkpoints/
├── roberta_pretrained/

Project Structure
roberta-from-scratch/
├── tokenizer/          # Trained tokenizer files
├── data/               # Dataset preparation scripts
├── *.py                # Training, testing, and utility scripts
├── requirements.txt
└── README.md

Training Summary

Vocabulary size: 50,000

Model size: ~45M parameters

Training objective: Masked Language Modeling (MLM)

Epochs trained: 1

Training completed successfully on GPU

Loss decreased from ~15 to ~5.8, confirming correct learning behavior.

Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

License

This project is released for educational and research purposes.


---

## Why THIS is the right README ✅

- Clean
- Standard
- Familiar to everyone
- No over-explaining
- Looks **professional**, not noisy
- Same structure as popular Python repos
