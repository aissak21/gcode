# InCoder


print("Loading model...")
from datasets import load_dataset

dataset = load_dataset("ablam/gcode")
print("Loaded model :)")

# Tokenize texts with same vocab as training the model
from transformers import AutoTokenizer

# Preserve casing info
tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B",use_fast=True,do_lower_case=False)

# Tokenize our texts.
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

print("Tokenizing our texts...")
# In batches and processes to speed it up.
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=64)
print("Tokenized our texts :)")

# Postprocess: 1
print("Postprocess 1: Removing columns...")
tokenized_datasets = tokenized_datasets.remove_columns(['text', 'token_type_ids'])
print("Removed columns :)")

# One percent
import datasets
train_dataset, train_1percent = tokenized_datasets['train'].train_test_split(test_size=0.01).values() #(41297976, 3)
test_dataset, test_1percent = tokenized_datasets['test'].train_test_split(test_size=0.01).values() #(417152, 3)
lm_split = datasets.DatasetDict({'train': train_1percent, 'test': test_1percent}) #{'train': (417152, 3), 'test': (62724, 3)}

# Postprocess: 2
# Labels = ground truth for the model to learn from.
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# And after
print("Postprocess 2: Batch nicely...")
tokenized_datasets_label = lm_split.map(group_texts, batched=True, num_proc=64)
print("Batched nicely :)")


# Postprocess: 3
print("Postprocess 3: To torch..")
tokenized_datasets_label.set_format('torch')
print("Tis torched :)")


# Training with Torch
train_dataset_label = tokenized_datasets_label["train"].shuffle(seed=42)
test_dataset_label = tokenized_datasets_label["test"].shuffle(seed=42)

# Dataloader
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

train_dataloader = DataLoader(train_dataset_label, shuffle=True, batch_size=8)
test_dataloader = DataLoader(test_dataset_label, batch_size=8)

# Activate model
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

import torch
import torch.nn as nn
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print("Training scheduled :)")

torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model) # parallel on multiple GPU'S

model.to(device)

print("Begin training :)")
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) #forward pass
        #print(outputs.__dict__) 
        loss = outputs.loss #compute the loss
        
        loss.mean().backward() #backward

        optimizer.step() #optimize
        lr_scheduler.step() #forward pass with new parameters
        optimizer.zero_grad() #zero out grad
        progress_bar.update(1) #move the bar

# import torch
print("Saving the model :)")
model.save_pretrained("./gcode-incoder-finetuned-model")

print("Voila!")




