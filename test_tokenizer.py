from tokenizers import ByteLevelBPETokenizer
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
)

text = "Transformers are powerful models for language understanding."
encoding = tokenizer.encode(text)

print("Tokens:")
print(encoding.tokens)

print("\nToken IDs:")
print(encoding.ids)
