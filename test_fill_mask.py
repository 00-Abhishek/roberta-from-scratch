from pathlib import Path
import tempfile

import pytest
from transformers import RobertaForMaskedLM, RobertaTokenizerFast


CHECKPOINT_DIR = Path("checkpoints/checkpoint-1000")
TOKENIZER_DIR = Path("tokenizer")


def test_trained_checkpoint_and_tokenizer_can_be_saved():
    if not CHECKPOINT_DIR.exists():
        pytest.skip(
            "Trained checkpoint is missing at checkpoints/checkpoint-1000; "
            "download or train the model before running this integration test."
        )
    if not TOKENIZER_DIR.exists():
        pytest.skip("Local tokenizer directory is missing; run train_tokenizer.py first.")

    model = RobertaForMaskedLM.from_pretrained(str(CHECKPOINT_DIR))
    tokenizer = RobertaTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

    with tempfile.TemporaryDirectory() as tmpdir:
        final_model_dir = Path(tmpdir) / "final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        assert (final_model_dir / "config.json").exists()
        assert (final_model_dir / "tokenizer_config.json").exists()
