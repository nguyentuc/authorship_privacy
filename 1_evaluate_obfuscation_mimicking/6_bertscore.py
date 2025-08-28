import os
import pandas as pd
import numpy as np
from datasets import load_from_disk
from bert_score import score
import matplotlib.pyplot as plt

def compute_bertscore_for_text_pairs():
    """
    Computes BERTScore metrics between original texts and their mimicked versions
    for different speakers in the speech dataset.
    
    BERTScore leverages pre-trained contextual embeddings from BERT to measure
    semantic similarity between text pairs, providing precision, recall, and F1 scores.
    """
    # Configuration
    root_path = '/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/speech/deepseek/with_user_metadata/'
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    speakers = ['obama', 'bush', 'trump']
    batch_size = 32
    
    # Dictionary to store results
    bertscore_results = {}
    
    # Load the dataset
    print("Loading speech dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Process each speaker
    for speaker in speakers:
        print(f"\n{'='*40}\nComputing BERTScore for {speaker.upper()}\n{'='*40}")
        
        # Get original texts
        author_dataset = dataset.filter(
            lambda example: example["style"] == speaker and len(example["text"].split()) > 50
        )['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        
        # Load mimicked texts
        mimicking_path = os.path.join(root_path, 'mimicking_from_original', f'{speaker}.csv')
        
        if not os.path.exists(mimicking_path):
            print(f"Warning: File not found - {mimicking_path}")
            continue
            
        df = pd.read_csv(mimicking_path)
        
        if 'Mimicking' not in df.columns:
            print(f"Warning: 'Mimicking' column not found in {mimicking_path}")
            continue
            
        mimicking_text = df['Mimicking'].tolist()
        
        # Ensure both lists have the same length
        min_length = min(len(original_text), len(mimicking_text))
        if min_length == 0:
            print(f"No data available for {speaker}")
            continue
            
        original_text = original_text[:min_length]
        mimicking_text = mimicking_text[:min_length]
        
        print(f"Computing BERTScore for {min_length} text pairs...")
        
        # Compute BERTScore metrics
        P, R, F1 = score(
            original_text, 
            mimicking_text, 
            lang="en", 
            batch_size=batch_size, 
            verbose=True
        )
        
        # Store results
        bertscore_results[speaker] = {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item(),
            'precision_std': P.std().item(),
            'recall_std': R.std().item(),
            'f1_std': F1.std().item(),
            'individual_f1': F1.tolist()  # Store individual scores for analysis
        }
        
        # Print results
        print(f"\nBERTScore Results for {speaker}:")
        print(f"  Precision: {P.mean():.4f} (±{P.std():.4f})")
        print(f"  Recall: {R.mean():.4f} (±{R.std():.4f})")
        print(f"  F1: {F1.mean():.4f} (±{F1.std():.4f})")
        
        # Optionally, analyze the distribution of scores
        print("\nF1 Score Distribution:")
        percentiles = [25, 50, 75]
        for p in percentiles:
            percentile_value = np.percentile(F1.numpy(), p)
            print(f"  {p}th percentile: {percentile_value:.4f}")
        
        # Find examples with highest and lowest scores
        if len(F1) > 0:
            highest_idx = F1.argmax().item()
            lowest_idx = F1.argmin().item()
            
            print("\nExample with highest BERTScore F1:")
            print(f"  Score: {F1[highest_idx]:.4f}")
            print(f"  Original: {original_text[highest_idx][:100]}...")
            print(f"  Mimicked: {mimicking_text[highest_idx][:100]}...")
            
            print("\nExample with lowest BERTScore F1:")
            print(f"  Score: {F1[lowest_idx]:.4f}")
            print(f"  Original: {original_text[lowest_idx][:100]}...")
            print(f"  Mimicked: {mimicking_text[lowest_idx][:100]}...")
    
    # Print comparative summary
    print("\n\nCOMPARATIVE SUMMARY OF BERTSCORE RESULTS:")
    print("=" * 60)
    print(f"{'Speaker':<10} | {'Precision':<15} | {'Recall':<15} | {'F1':<15}")
    print("-" * 60)
    for speaker, results in bertscore_results.items():
        print(f"{speaker:<10} | {results['precision']:.4f} ±{results['precision_std']:.4f} | "
              f"{results['recall']:.4f} ±{results['recall_std']:.4f} | "
              f"{results['f1']:.4f} ±{results['f1_std']:.4f}")
    
    # Create visualization of F1 score distributions
    try:
        plt.figure(figsize=(10, 6))
        for speaker, results in bertscore_results.items():
            plt.hist(results['individual_f1'], alpha=0.7, bins=20, label=speaker)
        
        plt.xlabel('BERTScore F1')
        plt.ylabel('Frequency')
        plt.title('Distribution of BERTScore F1 Scores by Speaker')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('bertscore_distribution.png')
        print("\nSaved F1 score distribution plot to 'bertscore_distribution.png'")
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    compute_bertscore_for_text_pairs()