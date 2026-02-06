from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# Paths
data_path = Path("data/wikitext_train.txt")
tokenizer_dir = Path("tokenizer")

tokenizer_dir.mkdir(exist_ok=True)

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer
tokenizer.train(
    files=str(data_path),
    vocab_size=50_000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# Save tokenizer files
tokenizer.save_model(str(tokenizer_dir))


print("Tokenizer trained and saved to ./tokenizer/")
