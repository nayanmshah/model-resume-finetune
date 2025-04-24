from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")

input_text = "Instruction: What are the candidate's main skills?\nContext: John is skilled in Python and ML."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))