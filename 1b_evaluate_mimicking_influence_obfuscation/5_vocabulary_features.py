import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def vocabulary_size_and_diversity(texts):
    """
    Calculate vocabulary statistics for a collection of texts.
    
    Args:
        texts: List of text samples
        
    Returns:
        Tuple containing (vocabulary size, average text length, lexical diversity)
    """
    if not texts:
        return 0, 0, 0
        
    # Tokenize texts and calculate lengths
    all_tokens = []
    text_lengths = []
    
    for text in texts:
        if not isinstance(text, str):
            continue
            
        tokens = word_tokenize(text.lower())
        text_lengths.append(len(tokens))
        all_tokens.extend(tokens)
    
    # Calculate statistics
    total_tokens = len(all_tokens)
    unique_tokens = set(all_tokens)
    vocab_size = len(unique_tokens)
    
    # Calculate average length and diversity
    avg_length = np.mean(text_lengths) if text_lengths else 0
    diversity = vocab_size / total_tokens if total_tokens > 0 else 0
    
    return vocab_size, avg_length, diversity


def analyze_vocabulary_statistics(data_name, api, with_metadata=True):
    """
    Analyze vocabulary statistics for original and obfuscated texts.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        with_metadata: Whether to use results with user metadata
    """
    # Set up paths
    metadata_str = "with_user_metadata" if with_metadata else "without_user_metadata"
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_str}/'
    
    # Load dataset
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset structure: {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Define speakers for the speech dataset
    speakers = ['obama', 'bush', 'trump']
    
    # Prepare results containers
    results = {
        'speaker': [],
        'text_type': [],
        'vocab_size': [],
        'avg_length': [],
        'diversity': []
    }
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nAnalyzing vocabulary statistics for {speaker}")
        
        # Get original text samples
        author_dataset = dataset.filter(
            lambda example: example["style"] == speaker and len(example["text"].split()) > 50
        )['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        
        # Calculate statistics for original text
        vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
        print(f"  Original: Vocab size: {vocab_size}, Avg length: {avg_length:.2f}, Diversity: {diversity:.4f}")
        
        # Store results
        results['speaker'].append(speaker)
        results['text_type'].append('original')
        results['vocab_size'].append(vocab_size)
        results['avg_length'].append(avg_length)
        results['diversity'].append(diversity)
        
        # Process obfuscation from correct attribute
        correct_path = os.path.join(root_path, 'obfuscation_from_correct_attribute', f'{speaker}.csv')
        if os.path.exists(correct_path):
            df = pd.read_csv(correct_path)
            obfuscation_text = df['Obfuscation'].tolist()
            
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text)
            print(f"  Obfuscation from correct: Vocab size: {vocab_size}, Avg length: {avg_length:.2f}, Diversity: {diversity:.4f}")
            
            # Store results
            results['speaker'].append(speaker)
            results['text_type'].append('obfuscation_correct')
            results['vocab_size'].append(vocab_size)
            results['avg_length'].append(avg_length)
            results['diversity'].append(diversity)
        else:
            print(f"  Warning: File not found - {correct_path}")
        
        # Process obfuscation from incorrect attribute
        incorrect_path = os.path.join(root_path, 'obfuscation_from_incorrect_attribute', f'{speaker}.csv')
        if os.path.exists(incorrect_path):
            df = pd.read_csv(incorrect_path)
            obfuscation_text = df['Obfuscation'].tolist()
            
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text)
            print(f"  Obfuscation from incorrect: Vocab size: {vocab_size}, Avg length: {avg_length:.2f}, Diversity: {diversity:.4f}")
            
            # Store results
            results['speaker'].append(speaker)
            results['text_type'].append('obfuscation_incorrect')
            results['vocab_size'].append(vocab_size)
            results['avg_length'].append(avg_length)
            results['diversity'].append(diversity)
        else:
            print(f"  Warning: File not found - {incorrect_path}")
    
    # Calculate average diversity for obfuscated texts
    obfuscation_diversities = []
    for speaker in speakers:
        speaker_results = pd.DataFrame(results)
        speaker_data = speaker_results[speaker_results['speaker'] == speaker]
        
        obfuscation_data = speaker_data[speaker_data['text_type'].isin(['obfuscation_correct', 'obfuscation_incorrect'])]
        if not obfuscation_data.empty:
            avg_obfuscation_diversity = obfuscation_data['diversity'].mean()
            obfuscation_diversities.append(avg_obfuscation_diversity)
            print(f"\nAverage obfuscation diversity for {speaker}: {avg_obfuscation_diversity:.4f}")
    
    if obfuscation_diversities:
        mean_diversity = np.mean(obfuscation_diversities)
        print(f"\nAverage diversity across all speakers: {mean_diversity:.4f}")
    
    # Create visualization
    create_vocabulary_plots(results, data_name, api, metadata_str)


def create_vocabulary_plots(results, data_name, api, metadata_str):
    """
    Create visualizations for vocabulary statistics.
    
    Args:
        results: Dictionary containing vocabulary statistics
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_str: String indicating whether user metadata was used
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create plots
        plt.figure(figsize=(18, 12))
        
        # 1. Vocabulary size comparison
        plt.subplot(2, 2, 1)
        sns.barplot(x='speaker', y='vocab_size', hue='text_type', data=df)
        plt.title('Vocabulary Size Comparison')
        plt.ylabel('Vocabulary Size')
        plt.xticks(rotation=45)
        
        # 2. Average text length comparison
        plt.subplot(2, 2, 2)
        sns.barplot(x='speaker', y='avg_length', hue='text_type', data=df)
        plt.title('Average Text Length Comparison')
        plt.ylabel('Average Tokens per Text')
        plt.xticks(rotation=45)
        
        # 3. Lexical diversity comparison
        plt.subplot(2, 2, 3)
        sns.barplot(x='speaker', y='diversity', hue='text_type', data=df)
        plt.title('Lexical Diversity Comparison')
        plt.ylabel('Lexical Diversity (Unique Tokens / Total Tokens)')
        plt.xticks(rotation=45)
        
        # 4. Diversity comparison across text types
        plt.subplot(2, 2, 4)
        sns.boxplot(x='text_type', y='diversity', data=df)
        plt.title('Lexical Diversity Distribution by Text Type')
        plt.ylabel('Lexical Diversity')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'vocabulary_statistics_{data_name}_{api}_{metadata_str}.png', dpi=300)
        print(f"\nSaved vocabulary statistics plot to vocabulary_statistics_{data_name}_{api}_{metadata_str}.png")
        
    except Exception as e:
        print(f"Error creating vocabulary plots: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'deepseek'
    
    # Analyze with user metadata
    analyze_vocabulary_statistics(data_name, api, with_metadata=True)