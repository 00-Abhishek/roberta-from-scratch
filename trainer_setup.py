import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Tokenizer
# -------------------------
tokenizer = RobertaTokenizerFast(
    vocab_file="tokenizer/vocab.json",
    merges_file="tokenizer/merges.txt",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

# -------------------------
# Dataset
# -------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# -------------------------
# Data collator (MLM)
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# -------------------------
# Model config + model
# -------------------------
config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=130,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config)

# -------------------------
# Training arguments (DRY RUN)
# -------------------------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=5e-4,
    max_steps=20,                 # DRY RUN
    logging_steps=5,
    save_steps=20,
    save_total_limit=1,
    report_to="none",
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Trainer initialized successfully")
