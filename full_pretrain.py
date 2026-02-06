import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# -------------------------
# CUDA sanity check
# -------------------------
assert torch.cuda.is_available(), "CUDA is not available."

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding=False,
    )


# -------------------------
# Tokenizer
# -------------------------
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

print("Tokenizer loaded")
print("Vocab size:", tokenizer.vocab_size)

# -------------------------
# Dataset (TRAIN ONLY)
# -------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )

train_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# 🔴 FILTER OUT EMPTY SEQUENCES
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids"]) > 0
)
# -------------------------
# MLM Data collator
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# -------------------------
# Model
# -------------------------
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=130,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config)

# -------------------------
# Training arguments (OLD API SAFE)
# -------------------------
training_args = TrainingArguments(
    output_dir="./roberta_pretrained",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=5e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    save_total_limit=3,
    report_to="none",
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# -------------------------
# Train
# -------------------------
trainer.train()

# -------------------------
# Save final model
# -------------------------
trainer.save_model("./roberta_pretrained")
tokenizer.save_pretrained("./roberta_pretrained")

print("Pretraining complete.")
