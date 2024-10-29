# Import necessary libraries
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import os
import random
import json
EPOCHS = 30

def load_dataset_k_user(data_path, k):
    train_texts, train_labels = [], []
    eval_texts, eval_labels = [], []
    test_texts, test_labels = [], []
    # train
    writing_files_train = os.listdir(data_path+'train/')
    random.seed(2024)
    writing_files_train = random.choices(writing_files_train, k=k)

    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train/'+ writing)
        for idx ,row  in writing_train.iterrows():
            train_texts.append(row['Answer'])
            train_labels.append(writing.split('.')[0])

    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'eval/'+ writing)
        for idx ,row  in writing_test.iterrows():
            eval_texts.append(row['Answer'])
            eval_labels.append(writing.split('.')[0])

        
    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'test/'+ writing)
        for idx ,row  in writing_test.iterrows():
            test_texts.append(row['Answer'])
            test_labels.append(writing.split('.')[0])

    return train_texts, train_labels, eval_texts, eval_labels ,test_texts, test_labels


# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)
        
# Evaluation function
def eval_model(model, data_loader, device):
    model = model.eval()
    correct_predictions = 0
    losses =[]
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    return float(correct_predictions) / len(data_loader.dataset), np.mean(losses)
    
for k in [10]:
    best_batchsize = None
    best_learningrate = None
    best_accuracy_eval = 0
    results = []

    # Train-test split
    train_data, train_labels, eval_data, eval_labels, test_data, test_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)

    # convert label
    labels_dictionary = {}
    idx = 0
    for label in train_labels:
        if label not in labels_dictionary:
            labels_dictionary[label] = idx
            idx+=1

    # encode new labels
    encoded_train_labels, encoded_eval_labels, encoded_test_labels = [],[], []
    for label in train_labels:
        encoded_train_labels.append(labels_dictionary[label])

    for label in eval_labels:
        encoded_eval_labels.append(labels_dictionary[label])

    for label in test_labels:
        encoded_test_labels.append(labels_dictionary[label])


    # Load BERT tokenizer and model for sequence classification
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))
    # Dataset and DataLoader setup
    train_dataset = TextDataset(train_data, encoded_train_labels, tokenizer, max_len=128)
    val_dataset = TextDataset(eval_data, encoded_eval_labels, tokenizer, max_len=128)
    test_dataset = TextDataset(test_data, encoded_test_labels, tokenizer, max_len=128)


    batch_size = [8, 16, 32, 64, 128, 256]
    learning_rates = [3e-4, 1e-4,1e-5, 2e-5, 5e-5, 3e-5]

    for bz in batch_size:
        for lr in learning_rates:
            train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
            eval_loader = DataLoader(val_dataset, batch_size=bz)
            test_loader = DataLoader(test_dataset, batch_size=bz)

            # Setup optimizer and learning rate scheduler
            # Training loop
            device = torch.device("cuda")
            model = model.to(device)

            optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
            total_steps = len(train_loader) * EPOCHS 
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) 

            patience = 2  # Number of epochs with no improvement after which training will stop
            best_val_loss = float('inf')  # Initialize best loss as infinity
            epochs_without_improvement = 0  # Tracks how many epochs since last improvement
            save_path = "/media/volume/arkai-lab-data-private/Coding/AA/Baselines/bert_weights/"  # Path to save the best model
            for epoch in range(EPOCHS):
                train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

                # evaluate on evaluation dataset
                eval_acc, eval_loss = eval_model(model, eval_loader, device)
                print(f"Epoch {epoch+1}/{EPOCHS}: Train loss: {train_loss}, Train accuracy: {train_acc * 100:.2f}, Eval accuracy: {eval_acc * 100:.2f}, Evaluation Loss: {eval_loss}")

                # Check for improvement in validation loss
                if  eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    epochs_without_improvement = 0  # Reset patience counter
                    torch.save(model.state_dict(), save_path+'bert.pth')  # Save the best model
                    print("Validation loss improved. Saving the model.")
                else:
                    epochs_without_improvement += 1  # Increment the counter
                    print(f"No improvement for {epochs_without_improvement} epoch(s).")

                # Stop training if there has been no improvement for 'patience' epochs
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

            # Load the best model for evaluation on test set
            model.load_state_dict(torch.load(save_path+'bert.pth'))

            # Evaluate the model on the evaluation dataset
            train_acc, _ = eval_model(model, train_loader, device)
            eval_acc, _ = eval_model(model, eval_loader, device)
            
            # Store results
            results.append({
                'batch_size': bz,
                'learning_rate': lr,
                'train_accuracy': train_acc,
                'eval_accuracy': eval_acc
            })

            # Update the best model if this one is better
            if eval_acc > best_accuracy_eval:
                best_accuracy_eval = eval_acc
                best_batchsize = bz
                best_learningrate = lr
                torch.save(model.state_dict(), save_path+'best_model_'+str(k)+'.pth')
    
    results_df = pd.DataFrame(results)
    print("Results:\n", results_df)
    print(f"Best batchsize: {best_batchsize}, learning rate:{best_learningrate} with eval accuracy: {best_accuracy_eval:.4f}")

    # train and evaluate model with best alpha on test set
    print("Evaluate with best hyper parameters:")
    # Load the best model for evaluation on test set
    model.load_state_dict(torch.load(save_path+'best_model_'+str(k)+'.pth'))

    # Evaluate the model on the test dataset
    test_acc, _ = eval_model(model, test_loader, device)
    print(f"BERT: Accuracy with {k} users: {test_acc * 100:.2f}%")
    # update the best hyper_parameters
    data ={'best_batchsize': best_batchsize, 'best_learningrate': best_learningrate, 'test_acc': test_acc}
    with open('/media/volume/arkai-lab-data-private/Coding/AA/Baselines/bert_weights/hyper_parameters_configuration_'+str(k)+'.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(80*"=")
    del model