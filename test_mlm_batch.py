from pathlib import Path

import pytest
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast


TOKENIZER_DIR = Path("tokenizer")
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
MERGES_FILE = TOKENIZER_DIR / "merges.txt"


def test_mlm_data_collator_masks_local_tokenizer_batch():
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

    texts = [
        "Transformers are powerful language models.",
        "Masked language modeling trains bidirectional encoders.",
        "RoBERTa removes next sentence prediction.",
        "Dynamic masking changes the masked tokens per batch.",
    ]
    tokenized = [tokenizer(text, truncation=True, max_length=32) for text in texts]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    batch = data_collator(tokenized)
    masked_count = (batch["labels"] != -100).sum().item()

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == len(texts)
    assert masked_count > 0
