from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")

# Save model
model.save_pretrained("resume-qa-model")
tokenizer.save_pretrained("resume-qa-model")

print("✅ Saved model and tokenizer to resume-qa-model/")
