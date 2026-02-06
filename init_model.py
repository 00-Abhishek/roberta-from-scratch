import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, RobertaConfig

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=128,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config)

# Parameter count
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

# Move to GPU (test only)
if torch.cuda.is_available():
    model = model.cuda()
    print("Model successfully moved to GPU")
