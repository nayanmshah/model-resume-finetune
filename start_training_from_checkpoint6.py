from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("resume-qa-model/checkpoint-6")
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")