from transformers import RobertaConfig
config= RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=130,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_heads=8,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    type_vocab_size=1,
    layer_norm_eps=1e-12,
)
print(config)
