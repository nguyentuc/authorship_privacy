import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from datasets import load_from_disk

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_perplexity(text, model, tokenizer, max_length=512):
    """
    Compute perplexity for a single text.
    
    Args:
        text: Input text to evaluate
        model: Language model (with LoRA weights)
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length to process
        
    Returns:
        Perplexity value (lower is better for matching the model's distribution)
    """
    # Handle empty or invalid inputs
    if not text or not isinstance(text, str):
        return float('inf')
    
    # Encode the text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids
    
    # Compute perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


def ppl_dif_between_3_datasets(dataset_name, api, with_without):
    """
    Compute perplexity on original, obfuscated, and mimicked text samples
    using author-specific fine-tuned language models.
    
    Args:
        dataset_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        with_without: Whether user metadata was used ('with_user_metadata' or 'without_user_metadata')
        
    Returns:
        Dictionary containing perplexity values for different text types and authors
    """
    # Set paths
    base_path = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/'
    
    # Load the base GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add the EOS token as PAD token to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dictionary to store all perplexity values
    text_level_perplexities = {}
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        authors = list(set(dataset['train']['style']))
        
        for person in authors:
            print(f"Computing perplexity for {person}'s model")
            
            # Load the LoRA weights for this author
            lora_weights_path = f'/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/{person}_16_32/lora_weights'
            
            if not os.path.exists(lora_weights_path):
                print(f"Warning: LoRA weights not found for {person}")
                continue
                
            # Load the author-specific model
            model = PeftModel.from_pretrained(base_model, lora_weights_path)
            model.eval()
            
            # Initialize results for this author
            text_level_perplexities[person] = {}
            
            # Process different text types
            
            # 1. Mimicking samples
            mimicking_path = os.path.join(base_path, 'micking_sample', f'{person}.csv')
            if os.path.exists(mimicking_path):
                df = pd.read_csv(mimicking_path)
                mimicking_ppls = [compute_perplexity(row['Mimicking'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['Mimicking'] = mimicking_ppls
                print(f"  Mimicking samples: {len(mimicking_ppls)} texts, Avg PPL: {np.mean(mimicking_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {mimicking_path}")
            
            # 2. Obfuscation from mimicking
            obf_mimic_path = os.path.join(base_path, 'obfuscation_from_mimic', f'{person}.csv')
            if os.path.exists(obf_mimic_path):
                df = pd.read_csv(obf_mimic_path)
                obf_mimic_ppls = [compute_perplexity(row['Obfuscation'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['obfuscation_from_mimicking'] = obf_mimic_ppls
                print(f"  Obfuscation from mimicking: {len(obf_mimic_ppls)} texts, Avg PPL: {np.mean(obf_mimic_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {obf_mimic_path}")
            
            # 3. Obfuscation from original
            obf_orig_path = os.path.join(base_path, 'obfuscation_from_original', f'{person}.csv')
            if os.path.exists(obf_orig_path):
                df = pd.read_csv(obf_orig_path)
                obf_orig_ppls = [compute_perplexity(row['Obfuscation'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['obfuscation_from_original'] = obf_orig_ppls
                print(f"  Obfuscation from original: {len(obf_orig_ppls)} texts, Avg PPL: {np.mean(obf_orig_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {obf_orig_path}")
            
            # 4. Original texts
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            orig_ppls = [compute_perplexity(text['text'], model, tokenizer) for text in author_dataset]
            text_level_perplexities[person]['original_text'] = orig_ppls
            print(f"  Original texts: {len(orig_ppls)} texts, Avg PPL: {np.mean(orig_ppls):.2f}")
            
            print(f"  {'-'*40}")
    
    elif dataset_name == 'quora':
        # Process Quora dataset
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        
        for filename in os.listdir(root_path):
            if not filename.endswith('.txt'):
                continue
                
            person = filename.split('.')[0]
            print(f"Computing perplexity for {person}'s model")
            
            # Load the LoRA weights for this author
            lora_weights_path = f'/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/{person}_16_16/lora_weights'
            
            if not os.path.exists(lora_weights_path):
                print(f"Warning: LoRA weights not found for {person}")
                continue
                
            # Load the author-specific model
            model = PeftModel.from_pretrained(base_model, lora_weights_path)
            model.eval()
            
            # Initialize results for this author
            text_level_perplexities[person] = {}
            
            # Process different text types
            
            # 1. Mimicking samples
            mimicking_path = os.path.join(base_path, 'micking_sample', f'{person}.csv')
            if os.path.exists(mimicking_path):
                df = pd.read_csv(mimicking_path)
                mimicking_ppls = [compute_perplexity(row['Mimicking'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['mimicking'] = mimicking_ppls
                print(f"  Mimicking samples: {len(mimicking_ppls)} texts, Avg PPL: {np.mean(mimicking_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {mimicking_path}")
            
            # 2. Obfuscation from mimicking
            obf_mimic_path = os.path.join(base_path, 'obfuscation_from_mimic', f'{person}.csv')
            if os.path.exists(obf_mimic_path):
                df = pd.read_csv(obf_mimic_path)
                obf_mimic_ppls = [compute_perplexity(row['Obfuscation'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['obfuscation_from_mimicking'] = obf_mimic_ppls
                print(f"  Obfuscation from mimicking: {len(obf_mimic_ppls)} texts, Avg PPL: {np.mean(obf_mimic_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {obf_mimic_path}")
            
            # 3. Obfuscation from original
            obf_orig_path = os.path.join(base_path, 'obfuscation_from_original', f'{person}.csv')
            if os.path.exists(obf_orig_path):
                df = pd.read_csv(obf_orig_path)
                obf_orig_ppls = [compute_perplexity(row['Obfuscation'], model, tokenizer) for _, row in df.iterrows()]
                text_level_perplexities[person]['obfuscation_from_original'] = obf_orig_ppls
                print(f"  Obfuscation from original: {len(obf_orig_ppls)} texts, Avg PPL: {np.mean(obf_orig_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {obf_orig_path}")
            
            # 4. Original texts
            writing_file = f'/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/{person}.csv'
            if os.path.exists(writing_file):
                author_dataset = pd.read_csv(writing_file)
                author_dataset = author_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
                author_dataset = author_dataset.sample(frac=0.2, random_state=42)
                
                orig_ppls = [compute_perplexity(text['Question']+' '+text['Answer'], model, tokenizer) 
                             for _, text in author_dataset.iterrows()]
                text_level_perplexities[person]['original_text'] = orig_ppls
                print(f"  Original texts: {len(orig_ppls)} texts, Avg PPL: {np.mean(orig_ppls):.2f}")
            else:
                print(f"  Warning: File not found - {writing_file}")
            
            print(f"  {'-'*40}")
    
    return text_level_perplexities


# Main execution code
if __name__ == "__main__":
    # Initialize result container
    all_ppl = {'quora': {}}
    
    # Process all combinations of API and metadata settings
    for api in ['4o-mini', 'o3-mini', 'deepseek', 'gemini']:
        all_ppl['quora'][api] = {}
        
        for with_without in ['with_user_metadata', 'without_user_metadata']:
            print(f"\n{'='*80}\nProcessing quora dataset with {api} API ({with_without})\n{'='*80}")
            
            # Compute perplexity scores
            ppl_results = ppl_dif_between_3_datasets(
                dataset_name='quora', 
                api=api, 
                with_without=with_without
            )
            
            # Store results
            all_ppl['quora'][api][with_without] = ppl_results
            
            # Save intermediate results in case of failure
            with open(f"/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/quora/ppl_logs_{api}_{with_without}.json", "w") as f:
                json.dump({api: {with_without: ppl_results}}, f, indent=4)
    
    # Save the complete results
    with open("/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/quora/ppl_logs.json", "w") as f:
        json.dump(all_ppl, f, indent=4)
    
    print("\nAll perplexity calculations completed and saved!")