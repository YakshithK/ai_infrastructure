from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("t5-small")

max_length = 512

ds = load_from_disk("data/processed/hf_train")
ds_test = load_from_disk("data/processed/hf_test")

def prep(example):
    input_text = example["clean_text"]
    target_text = example["target_text"]  # Make sure this column exists in your dataset
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds = ds.map(prep, remove_columns=list(ds.column_names), batched=False)
ds_test = ds_test.map(prep, remove_columns=list(ds_test.column_names), batched=False)

ds.save_to_disk("data/processed/hf_train_tok")
ds_test.save_to_disk("data/processed/hf_test_tok")