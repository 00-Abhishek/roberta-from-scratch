from transformers import (
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

print("Data collator initialized successfully")
