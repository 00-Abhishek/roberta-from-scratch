from transformers import TrainingArguments
from pathlib import Path
import shutil

# Output directory
output_dir = Path("./checkpoints")

# Explicit cleanup (required for older Transformers)
if output_dir.exists():
    shutil.rmtree(output_dir)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1_000,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

print("TrainingArguments initialized successfully")
