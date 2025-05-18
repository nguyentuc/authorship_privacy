import os
import torch
import argparse
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA on text generation datasets")
parser.add_argument('--dataset', type=str, default="wikitext", help="Dataset to use: wikitext, openai/webtext, bookcorpus, etc.")
parser.add_argument('--model_name', type=str, default="gpt2", help="Model name: gpt2, gpt2-medium, gpt2-large, or gpt2-xl")
parser.add_argument('--lora_r', type=int, default=16, help="LoRA attention dimension")
parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--data_percentage', type=float, default=1.0, help="Percentage of dataset to use for training (0.05 = 5%)")
parser.add_argument('--eval_text', type=str, default="The quick brown fox jumps over the lazy dog. Artificial intelligence has revolutionized many industries. Climate change remains a significant global challenge.", help="Text for perplexity evaluation")
args = parser.parse_args()

# Load speech dataset
dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
print(f"Dataset structure: {dataset}")
personalize_models = list(set(dataset['train']['style']))


# for each person, train and evaluate a gpt2 for their writing
for person in personalize_models:
    # Configuration
    MODEL_NAME = args.model_name

    # Create output directories
    BASE_OUTPUT_DIR = person+"_"+str(args.lora_r)+"_"+str(args.lora_alpha)
    CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
    LORA_WEIGHTS_DIR = os.path.join(BASE_OUTPUT_DIR, "lora_weights")
    LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")
    EVAL_DIR = os.path.join(BASE_OUTPUT_DIR, "evaluation")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LORA_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Other parameters
    DATA_PERCENTAGE = args.data_percentage
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_SEQ_LENGTH = args.max_seq_length
    EVAL_TEXT = args.eval_text

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Save the tokenizer for later use with the LoRA weights
    tokenizer.save_pretrained(os.path.join(BASE_OUTPUT_DIR, "tokenizer"))

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"],  # Attention modules in GPT-2
        bias="none",
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    print("Model with LoRA adapters:")
    model.print_trainable_parameters()  # Prints the percentage of trainable parameters

    # Save the LoRA configuration
    peft_config.save_pretrained(LORA_WEIGHTS_DIR)

    # Load dataset for only that author
    author_dataset = dataset.filter(lambda example: example["style"] == person)
    # merge train and test to get as much data as we can 
    merge_dataset = concatenate_datasets([author_dataset['train'], author_dataset['test']])
    print(merge_dataset)
    exit()

    # Process dataset
    def tokenize_function(examples):
        texts = examples['text']
        # Filter out empty texts
        texts = [text for text in texts if text and isinstance(text, str) and len(text.strip()) > 0]
        
        result = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        
        # For datasets that might have other structures, ensure we're only returning properly formatted examples
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply tokenization to training dataset
    tokenized_train_dataset = merge_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=merge_dataset.column_names,
        desc="Tokenizing training dataset",
    )

    # Apply tokenization to validation dataset
    tokenized_validation_dataset = author_dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=author_dataset['validation'].column_names,
        desc="Tokenizing validation dataset",
    )


    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We want to perform causal language modeling, not masked LM
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision training if available
        eval_steps=1000 if tokenized_validation_dataset else None,
        evaluation_strategy="steps" if tokenized_validation_dataset else "no",
        load_best_model_at_end=True,  # Automatically select best model
        metric_for_best_model="perplexity",  # Custom metric for best model
        greater_is_better=False,  # Lower perplexity is better
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save only the LoRA weights
    print(f"Saving LoRA weights to {LORA_WEIGHTS_DIR}")
    model.save_pretrained(LORA_WEIGHTS_DIR)

    # Save a metadata file with information about the training
    with open(os.path.join(BASE_OUTPUT_DIR, "training_info.txt"), "w") as f:
        f.write(f"Base model: {MODEL_NAME}\n")
        f.write(f"Dataset percentage used: {DATA_PERCENTAGE*100:.1f}%\n")
        f.write(f"LoRA rank (r): {LORA_R}\n")
        f.write(f"LoRA alpha: {LORA_ALPHA}\n")
        f.write(f"LoRA dropout: {LORA_DROPOUT}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Max sequence length: {MAX_SEQ_LENGTH}\n")

    # Example function to load the LoRA model for inference
    def load_lora_model_for_inference(base_model_name, lora_weights_path):
        """Load a pre-trained model with LoRA weights for inference"""
        # Load the base model
        base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        
        # Load the LoRA weights
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        
        return model


    def compute_perplexity(model, dataset):
        model.eval()
        losses = []
        
        for batch in dataset:
            inputs = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
            with torch.no_grad():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(mean_loss)
        return perplexity

    # Generate text with the fine-tuned model
    def generate_text(model, tokenizer, prompt, max_length=100):
        model.eval()
        model.to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    # Test the model with example prompts
    print("="*50)
    print("Testing text generation with fine-tuned model:")
    prompts = [
        "Once upon a time",
        "The research findings suggest that"
    ]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
        print("-"*50)