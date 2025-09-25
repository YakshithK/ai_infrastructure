from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

# Load some test samples
TEST_CSV = "data/processed/clean_articles.csv"
NUM_SAMPLES = 5

df = pd.read_csv(TEST_CSV)
available_samples = df["clean_text"].dropna()
sample_count = min(NUM_SAMPLES, len(available_samples))
samples = available_samples.sample(sample_count, random_state=42).tolist()

# Load tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_outputs(model, samples, tokenizer):
    outputs = []
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=64)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)
    return outputs

# 1. Base T5-small outputs
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
base_outputs = generate_outputs(base_model, samples, tokenizer)

# 2. Fine-tuned model outputs
finetuned_model_hub = "yakshithk/t5-small-baseline"
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model_hub)
finetuned_outputs = generate_outputs(finetuned_model, samples, tokenizer)

# Print comparison
for i, text in enumerate(samples):
    print(f"\nSample {i+1}:")
    print(f"Input: {text}")
    print(f"Base T5 Output: {base_outputs[i]}")
    print(f"Fine-tuned Output: {finetuned_outputs[i]}")
