from transformers import RobertaTokenizerFast
from pathlib import Path

model_dir = Path("roberta_pretrained/final").resolve()

tokenizer = RobertaTokenizerFast(
    vocab_file=str(model_dir / "vocab.json"),
    merges_file=str(model_dir / "merges.txt"),
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

# THIS is the missing step
tokenizer.save_pretrained(model_dir)

print("tokenizer.json generated successfully")
