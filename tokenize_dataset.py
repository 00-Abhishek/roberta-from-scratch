from datasets import load_dataset
from transformers import RobertaTokenizerFast

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

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print(tokenized_datasets)
