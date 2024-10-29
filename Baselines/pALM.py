# Import necessary libraries
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Load BERT tokenizer and model for masked language modeling
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Example dataset
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Bright birds fly swiftly in the blue sky.",
    "Python is a great programming language.",
    "Data science is amazing for predictive models.",
    "Natural language processing is a subfield of AI.",
    "He won the chess game after a tough battle."
]

# Train-test split
X_train, X_test = train_test_split(corpus, test_size=0.2, random_state=42)

# Dataset and DataLoader setup
train_dataset = TextDataset(X_train, tokenizer, max_len=128)
test_dataset = TextDataset(X_test, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop for fine-tuning BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Mask some tokens for masked language modeling (MLM)
        rand = torch.rand(input_ids.shape).to(device)
        mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * (input_ids != tokenizer.sep_token_id)
        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = tokenizer.mask_token_id
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training loss: {avg_train_loss}")

# saving model with the name of each author from training data


# evaluation: for each test sample, load model for 200 authors to evaluate on each, the prediction is the name of model that have lowest perplexity
#             if the name of that model is similar with the name of the testing sample increase 1 in the test data
# Pseudo-perplexity calculation
def calculate_perplexity(model, data_loader, tokenizer):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Calculate loss for each token in the input (mask one token at a time)
            for i in range(input_ids.size(1)):  # Iterate over each token in the sentence
                input_ids_clone = input_ids.clone()
                original_token_id = input_ids_clone[:, i]
                input_ids_clone[:, i] = tokenizer.mask_token_id  # Mask the i-th token
                
                outputs = model(input_ids=input_ids_clone, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute the loss for the masked token
                masked_token_loss = F.cross_entropy(logits[:, i, :], original_token_id, reduction='sum')
                total_loss += masked_token_loss.item()
                total_tokens += 1

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

# Calculate perplexity on the test set
perplexity = calculate_perplexity(model, test_loader, tokenizer)
print(f"Pseudo-Perplexity on the test set: {perplexity}")

# evaluation
