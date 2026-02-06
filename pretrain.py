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

# -------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------
tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")
tokenizer.model_max_length = 128

print("Tokenizer vocab size:", len(tokenizer))

# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# -------------------------------------------------
# 3. Tokenization (remove token_type_ids)
# -------------------------------------------------
def tokenize_function(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )
    enc.pop("token_type_ids", None)
    return enc

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print("Dataset tokenized successfully")

# -------------------------------------------------
# 4. Data collator
# -------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# -------------------------------------------------
# 5. Model configuration (CRITICAL FIX HERE)
# -------------------------------------------------
config = RobertaConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,  # ✅ FIXED (was 128)
    type_vocab_size=2,
)

# -------------------------------------------------
# 6. Model
# -------------------------------------------------
model = RobertaForMaskedLM(config)
model.resize_token_embeddings(len(tokenizer))

print("Model initialized")
print("Total parameters:", sum(p.numel() for p in model.parameters()))

# -------------------------------------------------
# 7. Training arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# -------------------------------------------------
# 8. Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -------------------------------------------------
# 9. Train
# -------------------------------------------------
print("Starting training...")
trainer.train()
