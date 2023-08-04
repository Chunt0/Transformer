from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load your dataset builder and construct your dataset as necessary.
# In this case we split the data into training data
dataset = load_dataset("squad")

# Data set must be tokenized. So first initialize the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

# Define a function to tokenize the examples
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)

# Tokenize the dataset with the max_length parameter
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,   # Set the batch size for tokenization
)

model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Define the TrainingArguments for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    logging_dir="./logs",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the validation set
results = trainer.evaluate()

# Print the evaluation results
print(results)
