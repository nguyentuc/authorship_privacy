import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import pandas as pd
import json
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_perplexity(text, model, tokenizer, max_length=512):
    """Compute perplexity for a single text."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


# compute ppl of original, obfuscation and mimicking dataset
def ppl_dif_between_3_datasets(dataset_name, api, with_without):
    if dataset_name =='speech':
        # Load model and tokenizer
        model_name = "gpt2"  # Replace with your fine-tuned model name if needed
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        base_model = GPT2LMHeadModel.from_pretrained(model_name)

        # load dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_models = list(set(dataset['train']['style']))

        text_level_perplexities = {}
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/{with_without}/'
        for person in personalize_models:
            print(f"Computing the PPL with {person}")
            # Load the LoRA weights of Obama LoRA
            lora_weights_path='/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/'+person+'_16_32/lora_weights'
            model = PeftModel.from_pretrained(base_model, lora_weights_path)
            model.eval()

            # Add the EOS token as PAD token to avoid warnings
            tokenizer.pad_token = tokenizer.eos_token

            # Load datasets by other authors
            text_level_perplexities[person] = {}


            # mimicking from correct
            df = pd.read_csv(synthesize_dataset+'mimicking_from_correct_attribute/'+person+'.csv')
            list_ppl2 = []
            for index, row in df.iterrows():
                ppl = compute_perplexity(row['Mimicking'], model, tokenizer)
                list_ppl2.append(ppl)
            text_level_perplexities[person]['mimicking_from_correct'] = list_ppl2
            

            # obfuscation from incorrect
            df = pd.read_csv(synthesize_dataset+'mimicking_from_incorrect_attribute/'+person+'.csv')
            list_ppl3 = []
            for index, row in df.iterrows():
                ppl = compute_perplexity(row['Mimicking'], model, tokenizer)
                list_ppl3.append(ppl)
            text_level_perplexities[person]['mimicking_from_incorrect'] = list_ppl3

            # original text
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))

            list_ppl4 = []
            for text in author_dataset:
                ppl = compute_perplexity(text['text'], model, tokenizer)
                list_ppl4.append(ppl)
            text_level_perplexities[person]['original_text'] = list_ppl4
        return text_level_perplexities
            
            
    elif dataset_name=='quora':
        # Load model and tokenizer
        model_name = "gpt2"  # Replace with your fine-tuned model name if needed
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/{with_without}/'
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        text_level_perplexities ={}
        for filename in os.listdir(root_path):
            person = filename.split('.')[0]
            text_level_perplexities[person] = {}

            lora_weights_path='/media/volume/tucnv/Coding/AA/evaluation_metric/4_finetune_gpt2_with_lora/'+person+'_16_16/lora_weights'
            model = PeftModel.from_pretrained(base_model, lora_weights_path)
            model.eval()

            # Add the EOS token as PAD token to avoid warnings
            tokenizer.pad_token = tokenizer.eos_token
        

            df = pd.read_csv(synthesize_dataset+'mimicking_from_correct_attribute/'+person+'.csv')
            list_ppl2 = []
            for index, row in df.iterrows():
                ppl = compute_perplexity(row['Mimicking'], model, tokenizer)
                list_ppl2.append(ppl)
            text_level_perplexities[person]['mimicking_from_correct'] = list_ppl2
            

            df = pd.read_csv(synthesize_dataset+'mimicking_from_incorrect_attribute/'+person+'.csv')
            list_ppl3 = []
            for index, row in df.iterrows():
                ppl = compute_perplexity(row['Mimicking'], model, tokenizer)
                list_ppl3.append(ppl)
            text_level_perplexities[person]['mimicking_from_incorrect'] = list_ppl3

            # load and compute ppl on original text
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)

            list_ppl4 = []
            for idx, text in author_dataset.iterrows():
                ppl = compute_perplexity(text['Question']+' '+text['Answer'], model, tokenizer)
                list_ppl4.append(ppl)
            text_level_perplexities[person]['original_text'] = list_ppl4
        return text_level_perplexities

all_ppl = {}
for api in ['4o-mini', 'o3-mini', 'deepseek', 'gemini']:
    all_ppl[api] ={}
    for with_without in ['with_user_metadata', 'without_user_metadata']:
        print(f"Working on {api}-{with_without}")

        all_ppl[api][with_without] ={}
        ppl = ppl_dif_between_3_datasets(dataset_name='quora', api = api, with_without=with_without)
        all_ppl[api][with_without] = ppl

with open("/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/quora/ppl_logs.json", "w") as f:
    json.dump(all_ppl, f, indent=4)  # indent=4 makes it pretty


# Speech

all_ppl = {}
for api in ['4o-mini', 'o3-mini', 'deepseek', 'gemini']:
    all_ppl[api] ={}
    for with_without in ['with_user_metadata', 'without_user_metadata']:
        print(f"Working on {api}-{with_without}")

        all_ppl[api][with_without] ={}
        ppl = ppl_dif_between_3_datasets(dataset_name='speech', api = api, with_without=with_without)
        all_ppl[api][with_without] = ppl

with open("/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/speech/ppl_logs.json", "w") as f:
    json.dump(all_ppl, f, indent=4)  # indent=4 makes it pretty

    