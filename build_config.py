from transformers import RobertaConfig, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,  # 4 × hidden_size (standard)
    max_position_embeddings=128,
    type_vocab_size=1,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

print(config)
