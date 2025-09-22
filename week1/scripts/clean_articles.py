import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv("data/raw/raw_articles.csv") 

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    return text.lower().strip()

df['clean_text'] = df['text'].apply(clean_text)
df.to_csv("data/processed/clean_articles.csv", index=False)

hf_dataset = Dataset.from_pandas(df[['clean_text']])
hf_dataset.save_to_disk("data/processed/hf_dataset")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
df["tokens"] = df["clean_text"].apply(lambda x: tokenizer.tokenize(x)[:128])
df.to_csv("data/processed/tokenized.csv", index=False)