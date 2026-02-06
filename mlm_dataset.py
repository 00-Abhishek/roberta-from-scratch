from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

print(dataset)
print("\nSample:")
print(dataset["train"][0]["text"])
