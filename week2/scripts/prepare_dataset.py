from datasets import load_dataset, Dataset
import pandas as pd

df = pd.read_csv("data\\processed\\clean_articles.csv")

# Add target_text column as a copy of clean_text
if "clean_text" in df.columns:
    df["target_text"] = df["clean_text"]

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Ensure target_text is present in both train and test
if "target_text" not in train.columns:
    train["target_text"] = train["clean_text"]
if "target_text" not in test.columns:
    test["target_text"] = test["clean_text"]

ds = Dataset.from_pandas(train)
ds_test = Dataset.from_pandas(test)

ds.save_to_disk("data/processed/hf_train")
ds_test.save_to_disk("data/processed/hf_test")