import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


# The steps are - define a pretrained checkpoint
# Initialize a tokenizer and a model for the task you are wishing to use
# 

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["The detective carefully examined the evidence, piecing together clues to solve the mysterious murder case.","The team of scientists conducted extensive experiments, analyzing data from various sources to validate their groundbreaking hypothesis."]

batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

batch["labels"] = torch.tensor([1, 1])

optimizer = torch.optim.AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

raw_datasets = load_dataset("glue", "mrpc")
raw_train = raw_datasets["train"]
print(raw_train[15])
print(raw_train[87])