from transformers import RobertaForMaskedLM, RobertaTokenizerFast

# Load model from local checkpoint ONLY
model = RobertaForMaskedLM.from_pretrained(
    "./checkpoints/checkpoint-1000",
    local_files_only=True,
)

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(
    "./tokenizer",
    local_files_only=True,
)

# Save consolidated final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("Final model saved successfully to ./final_model")
