from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast(
    vocab_file="tokenizer/vocab.json",
    merges_file="tokenizer/merges.txt",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

print("Data collator initialized")
