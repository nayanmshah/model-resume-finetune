from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
import torch

# Load tokenizer and model
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load your training dataset
data_files = {"train": "train.jsonl", "validation": "val.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# Preprocessing
def preprocess(example):
    print(f"🧪 Preprocessing example: {example['instruction'][:40]}...")
    input_text = f"Instruction: {example['instruction']}\nContext: {example['context']}"
    target_text = example["response"]

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    targets = tokenizer(
        target_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    labels = targets["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss
    inputs["labels"] = labels
    return inputs

# Tokenize and clean
tokenized_dataset = dataset.map(
    preprocess,
    batched=False,
    remove_columns=dataset["train"].column_names
)
tokenized_dataset.set_format("torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./resume-qa-model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="no",      # disables auto checkpointing
    fp16=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Train
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete. Saving model now...")

# ✅ Save final model
trainer.model.save_pretrained("./resume-qa-model", safe_serialization=False)
tokenizer.save_pretrained("./resume-qa-model")
print("✅ Model saved to ./resume-qa-model")