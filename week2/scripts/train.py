from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
from huggingface_hub import login

login("HF_KEY")

wandb.init(project="week2-finetune", name="t5-small-baselines")

model_name = 't5-small'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
train_ds = load_from_disk("data/processed/hf_train_tok")
test_ds = load_from_disk("data/processed/hf_test_tok")

training_args = TrainingArguments(
    output_dir="output/t5-small-baseline",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    learning_rate=5e-5,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    report_to="wandb",
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()
trainer.push_to_hub()