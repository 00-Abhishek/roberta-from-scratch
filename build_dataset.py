from datasets import load_dataset
from transformers import RobertaTokenizerFast

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print(tokenized_dataset)
print("Example keys:", tokenized_dataset[0].keys())
