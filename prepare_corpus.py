from datasets import load_dataset
from pathlib import Path

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

output_path = Path("data")
output_path.mkdir(exist_ok=True)

with open(output_path / "wikitext_train.txt", "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        text = item["text"].strip()
        if text:
            f.write(text + "\n")

print("Corpus saved to data/wikitext_train.txt")
