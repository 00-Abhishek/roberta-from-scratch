from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
)
import torch

tokenizer = RobertaTokenizerFast(
    vocab_file="tokenizer/vocab.json",
    merges_file="tokenizer/merges.txt",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

tokenized = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

batch = data_collator([tokenized[i] for i in range(4)])

print("Input IDs shape:", batch["input_ids"].shape)
print("Labels shape:", batch["labels"].shape)

mask_token_id = tokenizer.mask_token_id
masked_count = (batch["input_ids"] == mask_token_id).sum().item()

print("Number of masked tokens:", masked_count)
