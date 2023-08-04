from transformers import AutoTokenizer
from datasets import load_dataset


raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentence_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentence_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("Bilibili is the best website.", "Nacho cheesea all over the house?")
inputs = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

print(inputs)