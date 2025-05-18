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

import json


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load model and tokenizer
model_name = "gpt2"  # Replace with your fine-tuned model name if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
base_model = GPT2LMHeadModel.from_pretrained(model_name)

# load dataset
dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
print(f"Dataset structure: {dataset}")
personalize_models = list(set(dataset['train']['style']))


def compute_perplexity(text, model, tokenizer, max_length=512):
    """Compute perplexity for a single text."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

for person in personalize_models:
    print(f"Computing the PPL with {person}")
    # Load the LoRA weights of Obama LoRA
    lora_weights_path='/media/volume/tucnv/Coding/AA/evaluation_metric/finetune_gpt2_with_lora/'+person+'_16_32/lora_weights'
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()

    # Add the EOS token as PAD token to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets by other authors
    text_level_perplexities = {}
    sent_level_perplexities = {}

    for author_data in personalize_models:
        author_dataset = dataset.filter(lambda example: example["style"] == author_data)['train']

        # Compute perplexity for SST-2 (text level)
        text_level_perplexities[author_data] = []
        for example in author_dataset:
            ppl = compute_perplexity(example['text'], model, tokenizer)
            text_level_perplexities[author_data].append(ppl)


        # Split the text into sentences
        # sent_level_perplexities[author_data] = []
        # for example in author_dataset:
        #     sentences = re.split(r'[.!?]', example['text'])
        #     sentences = [s.strip() for s in sentences if s.strip()]
        #     for sentence in sentences:
        #         ppl = compute_perplexity(sentence, model, tokenizer)
        #         sent_level_perplexities[author_data].append(ppl)

    # Save dictionary as JSON file
    with open('/media/volume/tucnv/Coding/AA/evaluation_metric/finetune_gpt2_with_lora/ppl_logs/'+person+'.json', 'w') as file:
        json.dump(text_level_perplexities, file, indent=4)

    # with open('/media/volume/tucnv/Coding/AA/finetune_gpt2_with_lora/ppl/'+person+'_sent.json', 'w') as file:
    #     json.dump(sent_level_perplexities, file, indent=4)

    print("Plotting visualization for: ", person)
    colors = ['skyblue', 'lightcoral', 'mediumseagreen']

    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # Plot all three distributions on one plot: text perplexity
    for i, (name, data) in enumerate(text_level_perplexities.items()):
        sns.histplot(data, kde=True, bins=50, color=colors[i], label=name, ax=axs[0])

    axs[0].set_xlabel('Perplexity')
    axs[0].set_ylabel('Proportion')
    axs[0].set_xlim(0, 100)
    axs[0].set_ylim(0, 300)
    axs[0].set_title('Text Level Perplexity')
    axs[0].legend()

    # Plot all three distributions on one plot: text perplexity
    # for i, (name, data) in enumerate(sent_level_perplexities.items()):
    #     sns.histplot(data, kde=True, bins=50, color=colors[i], label=name, ax =axs[1])

    # axs[1].set_xlabel('Perplexity')
    # axs[1].set_title('Sentence Level Perplexity')
    # axs[1].set_xlim(0, 100)
    # axs[1].set_ylim(0, 300)
    # axs[1].legend()


    plt.legend()
    plt.tight_layout()
    plt.savefig('perplexity_dist_'+person+'.png', dpi=300, bbox_inches='tight')
    plt.show()
