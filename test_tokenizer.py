from pathlib import Path
import sys

import pytest
from tokenizers import ByteLevelBPETokenizer


TOKENIZER_DIR = Path("tokenizer")
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
MERGES_FILE = TOKENIZER_DIR / "merges.txt"


def test_byte_level_bpe_tokenizer_encodes_text():
    if not VOCAB_FILE.exists() or not MERGES_FILE.exists():
        pytest.skip("Local tokenizer artifacts are missing; run train_tokenizer.py first.")

    # Force UTF-8 output on Windows when this file is run directly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tokenizer = ByteLevelBPETokenizer(str(VOCAB_FILE), str(MERGES_FILE))
    encoding = tokenizer.encode(
        "Transformers are powerful models for language understanding."
    )

    assert encoding.tokens
    assert encoding.ids
    assert len(encoding.tokens) == len(encoding.ids)
