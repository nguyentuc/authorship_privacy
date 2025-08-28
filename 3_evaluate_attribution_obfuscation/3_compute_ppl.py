import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm

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
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return float('inf')  # Return infinity for failed computations


def ppl_dif_between_3_datasets(dataset_name, api, with_without):
    """
    Compute perplexity on original, and obfuscated text samples
    using author-specific fine-tuned language models.
    
    Args:
        dataset_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        with_without: Whether user metadata was used ('with_user_metadata' or 'without_user_metadata')
        
    Returns:
        Dictionary containing perplexity values for different text types and authors
    """
    # Set paths
    base_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset_name}/{api}/{with_without}/'
    
    # Load base GPT-2 model and tokenizer
    model_name = "gpt2"
    print(f"Loading base model: {model_name}")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        # Add the EOS token as PAD token to avoid warnings
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    # Dictionary to store perplexity values
    text_level_perplexities = {}
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nComputing perplexity for {person}\n{'='*40}")
            
            # Load the LoRA weights for this author
            lora_weights_path = f'/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/{person}_16_32/lora_weights'
            
            if not os.path.exists(lora_weights_path):
                print(f"Warning: LoRA weights not found at {lora_weights_path}")
                continue
                
            # Load the author-specific model
            try:
                model = PeftModel.from_pretrained(base_model, lora_weights_path)
                model.eval()
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                continue
            
            # Initialize results for this author
            text_level_perplexities[person] = {}
            
            # Process each text type
            
            # 1. Obfuscation from correct attribute
            correct_path = os.path.join(base_path, 'obfuscation_from_correct_attribute', f'{person}.csv')
            if os.path.exists(correct_path):
                df = pd.read_csv(correct_path)
                correct_ppls = []
                
                print(f"Computing perplexity for obfuscation from correct attribute ({len(df)} samples)")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    ppl = compute_perplexity(row['Obfuscation'], model, tokenizer)
                    correct_ppls.append(ppl)
                
                text_level_perplexities[person]['obfuscation_from_correct'] = correct_ppls
                print(f"  Average PPL: {np.mean(correct_ppls):.2f}")
                
                # Save intermediate results
                with open(f"/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset_name}/{api}/{with_without}/ppl_logs/{person}_obfuscation_from_correct.json", "w") as f:
                    json.dump(correct_ppls, f)
            else:
                print(f"Warning: File not found - {correct_path}")
            
            # 2. Obfuscation from incorrect attribute
            incorrect_path = os.path.join(base_path, 'obfuscation_from_incorrect_attribute', f'{person}.csv')
            if os.path.exists(incorrect_path):
                df = pd.read_csv(incorrect_path)
                incorrect_ppls = []
                
                print(f"Computing perplexity for obfuscation from incorrect attribute ({len(df)} samples)")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    ppl = compute_perplexity(row['Obfuscation'], model, tokenizer)
                    incorrect_ppls.append(ppl)
                
                text_level_perplexities[person]['obfuscation_from_incorrect'] = incorrect_ppls
                print(f"  Average PPL: {np.mean(incorrect_ppls):.2f}")
                
                # Save intermediate results
                with open(f"/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset_name}/{api}/{with_without}/ppl_logs/{person}_obfuscation_from_incorrect.json", "w") as f:
                    json.dump(incorrect_ppls, f)
            else:
                print(f"Warning: File not found - {incorrect_path}")
            
            # 3. Original text
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            original_ppls = []
            print(f"Computing perplexity for original texts ({len(author_dataset)} samples)")
            for text in tqdm(author_dataset):
                ppl = compute_perplexity(text['text'], model, tokenizer)
                original_ppls.append(ppl)
            
            text_level_perplexities[person]['original_text'] = original_ppls
            print(f"  Average PPL: {np.mean(original_ppls):.2f}")
            
            # Save intermediate results
            with open(f"/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset_name}/{api}/{with_without}/ppl_logs/{person}_original.json", "w") as f:
                json.dump(original_ppls, f)
    
    elif dataset_name == 'quora':
        # Process Quora dataset
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        # Create directory for storing perplexity logs
        ppl_logs_dir = f"/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset_name}/{api}/{with_without}/ppl_logs/"
        os.makedirs(ppl_logs_dir, exist_ok=True)
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            person = filename.split('.')[0]
            print(f"\n{'='*40}\nComputing perplexity for {person}\n{'='*40}")
            
            # Load the LoRA weights for this author
            lora_weights_path = f'/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/{person}_16_16/lora_weights'
            
            if not os.path.exists(lora_weights_path):
                print(f"Warning: LoRA weights not found at {lora_weights_path}")
                continue
                
            # Load the author-specific model
            try:
                model = PeftModel.from_pretrained(base_model, lora_weights_path)
                model.eval()
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                continue
            
            # Initialize results for this author
            text_level_perplexities[person] = {}
            
            # Process each text type
            
            # 1. Obfuscation from correct attribute
            correct_path = os.path.join(base_path, 'obfuscation_from_correct_attribute', f'{person}.csv')
            if os.path.exists(correct_path):
                df = pd.read_csv(correct_path)
                correct_ppls = []
                
                print(f"Computing perplexity for obfuscation from correct attribute ({len(df)} samples)")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    ppl = compute_perplexity(row['Obfuscation'], model, tokenizer)
                    correct_ppls.append(ppl)
                
                text_level_perplexities[person]['obfuscation_from_correct'] = correct_ppls
                print(f"  Average PPL: {np.mean(correct_ppls):.2f}")
                
                # Save intermediate results
                with open(os.path.join(ppl_logs_dir, f"{person}_obfuscation_from_correct.json"), "w") as f:
                    json.dump(correct_ppls, f)
            else:
                print(f"Warning: File not found - {correct_path}")
            
            # 2. Obfuscation from incorrect attribute
            incorrect_path = os.path.join(base_path, 'obfuscation_from_incorrect_attribute', f'{person}.csv')
            if os.path.exists(incorrect_path):
                df = pd.read_csv(incorrect_path)
                incorrect_ppls = []
                
                print(f"Computing perplexity for obfuscation from incorrect attribute ({len(df)} samples)")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    ppl = compute_perplexity(row['Obfuscation'], model, tokenizer)
                    incorrect_ppls.append(ppl)
                
                text_level_perplexities[person]['obfuscation_from_incorrect'] = incorrect_ppls
                print(f"  Average PPL: {np.mean(incorrect_ppls):.2f}")
                
                # Save intermediate results
                with open(os.path.join(ppl_logs_dir, f"{person}_obfuscation_from_incorrect.json"), "w") as f:
                    json.dump(incorrect_ppls, f)
            else:
                print(f"Warning: File not found - {incorrect_path}")
            
            # 3. Original text
            writing_file = os.path.join(writing_dir, f"{person}.csv")
            if os.path.exists(writing_file):
                author_dataset = pd.read_csv(writing_file)
                author_dataset = author_dataset.sample(frac=0.4, random_state=42)
                
                original_ppls = []
                print(f"Computing perplexity for original texts ({len(author_dataset)} samples)")
                for idx, text in tqdm(author_dataset.iterrows(), total=len(author_dataset)):
                    ppl = compute_perplexity(text['Question'] + ' ' + text['Answer'], model, tokenizer)
                    original_ppls.append(ppl)
                
                text_level_perplexities[person]['original_text'] = original_ppls
                print(f"  Average PPL: {np.mean(original_ppls):.2f}")
                
                # Save intermediate results
                with open(os.path.join(ppl_logs_dir, f"{person}_original.json"), "w") as f:
                    json.dump(original_ppls, f)
            else:
                print(f"Warning: File not found - {writing_file}")
    
    return text_level_perplexities


# Main execution code
if __name__ == "__main__":
    # Dictionary to store all perplexity results
    all_ppl = {}
    
    # Process different API models and metadata settings
    for api in ['4o-mini', 'o3-mini', 'deepseek', 'gemini']:
        all_ppl[api] = {}
        
        for with_without in ['with_user_metadata', 'without_user_metadata']:
            print(f"\n\n{'='*80}\nWorking on {api} - {with_without}\n{'='*80}")
            
            # Create directory for logs
            os.makedirs(f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/quora/{api}/{with_without}/ppl_logs/', exist_ok=True)
            
            # Compute perplexity for this configuration
            try:
                ppl = ppl_dif_between_3_datasets(dataset_name='quora', api=api, with_without=with_without)
                all_ppl[api][with_without] = ppl
                
                # Save intermediate results for this configuration
                with open(f"/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/quora/{api}_{with_without}_ppl_logs.json", "w") as f:
                    json.dump({api: {with_without: ppl}}, f, indent=4)
            except Exception as e:
                print(f"Error processing {api} - {with_without}: {e}")
    
    # Save the complete results
    with open("/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/quora/ppl_logs.json", "w") as f:
        json.dump(all_ppl, f, indent=4)
    
    print("\nAll perplexity computations completed!")