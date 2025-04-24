from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained("./resume-qa-model")
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")

# Push to Hugging Face Hub
model.push_to_hub("nayanmshah/resume-qa-model")
tokenizer.push_to_hub("nayanmshah/resume-qa-model")
