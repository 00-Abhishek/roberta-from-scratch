import torch
from pathlib import Path
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    pipeline,
)

model_dir = Path("roberta_pretrained/final").resolve()
print("Loading model from:", model_dir)

tokenizer = RobertaTokenizerFast.from_pretrained(
    model_dir,
    local_files_only=True,
)

model = RobertaForMaskedLM.from_pretrained(
    model_dir,
    local_files_only=True,
)

device = 0 if torch.cuda.is_available() else -1

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

print("Model and tokenizer loaded successfully")

sentences = [
    "Natural language processing is a <mask> field of study.",
    "Transformers are very <mask> models for language tasks.",
    "The purpose of a neural network is to <mask> patterns in data.",
]

for s in sentences:
    print(f"\nInput: {s}")
    outputs = fill_mask(s, top_k=5)
    for o in outputs:
        print(f"  {o['token_str']:<12} | score={o['score']:.4f}")
