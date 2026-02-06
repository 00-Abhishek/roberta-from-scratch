from transformers import RobertaForMaskedLM, RobertaTokenizerFast

model = RobertaForMaskedLM.from_pretrained("./checkpoints/checkpoint-1000")
tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("Final model saved to ./final_model")
