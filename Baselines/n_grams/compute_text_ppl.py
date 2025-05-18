import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os
import pandas as pd 
import json

def compute_gpt2_perplexity(text, model_name='gpt2'):
    """
    Compute perplexity of a text using GPT-2
    
    Parameters:
    - text: Input text to evaluate
    - model_name: GPT-2 model variant (default: 'gpt2')
    
    Returns:
    - Perplexity score
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize the input text
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    
    # Compute sequence length
    seq_len = encodings.input_ids.size(1)
    
    # Prepare for perplexity computation
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Compute perplexity
    with torch.no_grad():
        # Move inputs to the same device as the model
        input_ids = encodings.input_ids.to(device)
        labels = input_ids.clone().to(device)
        
        # Get model outputs
        outputs = model(input_ids, labels=labels)
        
        # Compute loss for each token
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        # Reshape loss to match input sequence
        loss = loss.view(input_ids.size(0), -1)
        
        # Compute mean loss
        mean_loss = loss.mean().item()
        
        # Compute perplexity
        perplexity = np.exp(mean_loss)
    
    return perplexity

def example_gpt2_perplexity():
    # For each author compute PPL of the original/synthesize writing text and provide mean and std
    ppl_original ={}
    all_files = os.listdir('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/train_40_5_5/')
    for writing in all_files:
        print(f"Working on {writing.split('.')[0]}")
        ppl_each_user =[]
        synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/train_40_5_5/'+ writing)
        for idx ,row  in synthesize_writing_only.iterrows():
            perplexity = compute_gpt2_perplexity(row['Answer'])
            ppl_each_user.append(perplexity)

        # compute mean and std
        mean = np.mean(ppl_each_user)
        std = np.std(ppl_each_user)
        ppl_original[writing.split('.')[0]] = [mean, std]
    with open("/media/volume/arkai-lab-data-private/Coding/AA/Baselines/n_grams/ppl_logs/original.json", "w") as json_file:
        json.dump(ppl_original, json_file, indent=4)   

# Uncomment to run example
example_gpt2_perplexity()

def advanced_perplexity_analysis(text, model_name='gpt2'):
    """
    Advanced perplexity analysis with more detailed breakdown
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize the input text
    encodings = tokenizer(text, return_tensors='pt')
    
    # Compute per-token perplexity
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        input_ids = encodings.input_ids.to(device)
        labels = input_ids.clone().to(device)
        
        outputs = model(input_ids, labels=labels)
        
        # Compute loss for each token
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        # Decode tokens for analysis
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Create detailed analysis
        token_analysis = []
        for token, token_loss in zip(tokens, loss[0]):
            token_analysis.append({
                'token': token,
                'loss': token_loss.item(),
                'perplexity': np.exp(token_loss.item())
            })
        
        return token_analysis

# text = "The quick brown fox jumps over the lazy dog."
# analysis = advanced_perplexity_analysis(text)
# for item in analysis:
#     print(f"Token: {item['token']}, Loss: {item['loss']:.4f}, Token Perplexity: {item['perplexity']:.4f}")