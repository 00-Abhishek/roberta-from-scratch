import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, RobertaConfig

tokenizer=RobertaTokenizerFast(
    vocab_file='tokenizer/vocab.json',
    merges_file='tokenizer/merges.txt',
    bos_token='<s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>',

)
config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=130,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    type_vocab_size=1,
)

model=RobertaForMaskedLM(config)
model.eval()

inputs= tokenizer(
    'Transformers are <mask> models.',
    return_tensors='pt'
)

with torch.no_grad():
    outputs=model(**inputs)

print('Logits shape:', outputs.logits.shape)
