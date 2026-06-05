from pathlib import Path

import pytest
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast


TOKENIZER_DIR = Path("tokenizer")
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
MERGES_FILE = TOKENIZER_DIR / "merges.txt"


def _tiny_roberta_config(vocab_size=32):
    return RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        type_vocab_size=1,
    )


def test_roberta_forward_pass_with_synthetic_inputs():
    model = RobertaForMaskedLM(_tiny_roberta_config())
    model.eval()

    inputs = {
        "input_ids": torch.tensor([[0, 5, 6, 7, 2]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
    }

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape == (1, 5, 32)


def test_roberta_forward_pass_with_local_tokenizer():
    if not VOCAB_FILE.exists() or not MERGES_FILE.exists():
        pytest.skip("Local tokenizer artifacts are missing; run train_tokenizer.py first.")

    tokenizer = RobertaTokenizerFast(
        vocab_file=str(VOCAB_FILE),
        merges_file=str(MERGES_FILE),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    model = RobertaForMaskedLM(_tiny_roberta_config(vocab_size=tokenizer.vocab_size))
    model.eval()

    inputs = tokenizer("Transformers are <mask> models.", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape[:2] == inputs["input_ids"].shape
