from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(
    "./tokenizer",
    max_len=128,
)

print("Vocab size:", tokenizer.vocab_size)

# Quick sanity check
encoded = tokenizer(
    "Transformers are powerful models for language understanding.",
    return_tensors="pt",
)

print("Input IDs shape:", encoded["input_ids"].shape)
