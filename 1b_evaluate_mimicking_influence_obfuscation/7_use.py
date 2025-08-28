import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import tensorflow as tf
import tensorflow_hub as hub

def calculate_semantic_similarity(data_name, api, metadata_setting="with_user_metadata"):
    """
    Calculate semantic similarity between original and obfuscated texts using 
    Universal Sentence Encoder embeddings.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        metadata_setting: Setting to use ("with_user_metadata" or "without_user_metadata")
        
    Returns:
        Dictionary containing similarity metrics for each speaker
    """
    # Set paths
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_setting}/'
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    
    print(f"Loading Universal Sentence Encoder...")
    # Load the Universal Sentence Encoder model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Load dataset
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset structure: {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}
    
    # Define speakers
    speakers = ['obama', 'bush', 'trump']
    
    # Dictionary to store similarity metrics
    similarity_metrics = {}
    all_similarities = {
        'correct': [],
        'incorrect': []
    }
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nAnalyzing semantic similarity for {speaker}")
        similarity_metrics[speaker] = {}
        
        # Get original texts
        author_dataset = dataset.filter(
            lambda example: example["style"] == speaker and len(example["text"].split()) > 50
        )['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        
        # Load obfuscation from correct attribute
        correct_path = os.path.join(root_path, 'obfuscation_from_correct_attribute', f'{speaker}.csv')
        if not os.path.exists(correct_path):
            print(f"Warning: File not found - {correct_path}")
            continue
            
        df = pd.read_csv(correct_path)
        obfuscation_correct = df['Obfuscation'].tolist()
        
        # Load obfuscation from incorrect attribute
        incorrect_path = os.path.join(root_path, 'obfuscation_from_incorrect_attribute', f'{speaker}.csv')
        if not os.path.exists(incorrect_path):
            print(f"Warning: File not found - {incorrect_path}")
            continue
            
        df = pd.read_csv(incorrect_path)
        obfuscation_incorrect = df['Obfuscation'].tolist()
        
        # Ensure all lists have the same length for comparison
        min_length = min(len(original_text), len(obfuscation_correct), len(obfuscation_incorrect))
        original_text = original_text[:min_length]
        obfuscation_correct = obfuscation_correct[:min_length]
        obfuscation_incorrect = obfuscation_incorrect[:min_length]
        
        print(f"  Processing {min_length} texts for each category")
        
        # Compute embeddings with Universal Sentence Encoder
        print("  Computing embeddings...")
        try:
            original_embeddings = model(original_text)
            obfuscation_correct_embeddings = model(obfuscation_correct)
            obfuscation_incorrect_embeddings = model(obfuscation_incorrect)
            
            # Normalize embeddings for cosine similarity
            norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
            norm_obfuscation_correct_embeddings = tf.nn.l2_normalize(obfuscation_correct_embeddings, axis=1)
            norm_obfuscation_incorrect_embeddings = tf.nn.l2_normalize(obfuscation_incorrect_embeddings, axis=1)
            
            # Compute similarity matrices
            # Original vs correct obfuscation
            cosine_sim_matrix_correct = tf.matmul(
                norm_original_embeddings, 
                norm_obfuscation_correct_embeddings, 
                transpose_b=True
            )
            
            # Extract diagonal elements (pairwise comparisons)
            diagonal_similarities_correct = tf.linalg.diag_part(cosine_sim_matrix_correct)
            mean_similarity_correct = tf.reduce_mean(diagonal_similarities_correct)
            
            # Original vs incorrect obfuscation
            cosine_sim_matrix_incorrect = tf.matmul(
                norm_original_embeddings, 
                norm_obfuscation_incorrect_embeddings, 
                transpose_b=True
            )
            
            diagonal_similarities_incorrect = tf.linalg.diag_part(cosine_sim_matrix_incorrect)
            mean_similarity_incorrect = tf.reduce_mean(diagonal_similarities_incorrect)
            
            # Store results
            similarity_metrics[speaker]['use_correct'] = mean_similarity_correct.numpy()
            similarity_metrics[speaker]['use_incorrect'] = mean_similarity_incorrect.numpy()
            similarity_metrics[speaker]['use_diag_correct'] = diagonal_similarities_correct.numpy().tolist()
            similarity_metrics[speaker]['use_diag_incorrect'] = diagonal_similarities_incorrect.numpy().tolist()
            
            # Accumulate for overall averages
            all_similarities['correct'].extend(diagonal_similarities_correct.numpy())
            all_similarities['incorrect'].extend(diagonal_similarities_incorrect.numpy())
            
            print(f"  Semantic similarity (USE) - Original vs Correct: {mean_similarity_correct.numpy():.4f}")
            print(f"  Semantic similarity (USE) - Original vs Incorrect: {mean_similarity_incorrect.numpy():.4f}")
            print(f"  Average semantic similarity: {(mean_similarity_correct.numpy() + mean_similarity_incorrect.numpy()) / 2:.4f}")
            
        except Exception as e:
            print(f"  Error computing embeddings: {e}")
    
    # Calculate overall averages
    if all_similarities['correct'] and all_similarities['incorrect']:
        overall_avg_correct = np.mean(all_similarities['correct'])
        overall_avg_incorrect = np.mean(all_similarities['incorrect'])
        overall_avg = (overall_avg_correct + overall_avg_incorrect) / 2
        
        print(f"\nOverall average semantic similarity (correct): {overall_avg_correct:.4f}")
        print(f"Overall average semantic similarity (incorrect): {overall_avg_incorrect:.4f}")
        print(f"Overall average semantic similarity: {overall_avg:.4f}")
    
    # Create visualizations
    create_similarity_plots(similarity_metrics, all_similarities, data_name, api, metadata_setting)
    
    return similarity_metrics


def create_similarity_plots(similarity_metrics, all_similarities, data_name, api, metadata_setting):
    """
    Create visualizations for semantic similarity metrics.
    
    Args:
        similarity_metrics: Dictionary containing similarity metrics by speaker
        all_similarities: Dictionary containing all similarity scores
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: Setting used
    """
    try:
        # 1. Bar plot of average similarities by speaker
        plt.figure(figsize=(12, 6))
        
        speakers = list(similarity_metrics.keys())
        correct_values = [similarity_metrics[s]['use_correct'] for s in speakers]
        incorrect_values = [similarity_metrics[s]['use_incorrect'] for s in speakers]
        
        x = np.arange(len(speakers))
        width = 0.35
        
        plt.bar(x - width/2, correct_values, width, label='Correct Attribution')
        plt.bar(x + width/2, incorrect_values, width, label='Incorrect Attribution')
        
        plt.xlabel('Speaker')
        plt.ylabel('Semantic Similarity (USE)')
        plt.title(f'USE Semantic Similarity: Original vs Obfuscated Texts ({api})')
        plt.xticks(x, [s.capitalize() for s in speakers])
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'use_similarity_{data_name}_{api}_{metadata_setting}.png', dpi=300)
        
        # 2. Distribution plots
        plt.figure(figsize=(15, 10))
        
        # Create distribution plots for each speaker
        for i, speaker in enumerate(similarity_metrics.keys()):
            plt.subplot(2, len(similarity_metrics), i+1)
            
            correct_diag = similarity_metrics[speaker]['use_diag_correct']
            incorrect_diag = similarity_metrics[speaker]['use_diag_incorrect']
            
            sns.histplot(correct_diag, kde=True, bins=15, alpha=0.6, label='Correct Attribution')
            sns.histplot(incorrect_diag, kde=True, bins=15, alpha=0.6, label='Incorrect Attribution')
            
            plt.title(f'{speaker.capitalize()}: Semantic Similarity Distribution')
            plt.xlabel('Semantic Similarity (USE)')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Add vertical lines for means
            plt.axvline(np.mean(correct_diag), color='blue', linestyle='--', alpha=0.7)
            plt.axvline(np.mean(incorrect_diag), color='orange', linestyle='--', alpha=0.7)
        
        # Create overall distribution plot
        plt.subplot(2, 1, 2)
        sns.histplot(all_similarities['correct'], kde=True, bins=20, alpha=0.6, label='Correct Attribution')
        sns.histplot(all_similarities['incorrect'], kde=True, bins=20, alpha=0.6, label='Incorrect Attribution')
        
        plt.title('Overall Semantic Similarity Distribution')
        plt.xlabel('Semantic Similarity (USE)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Add vertical lines for means
        plt.axvline(np.mean(all_similarities['correct']), color='blue', linestyle='--', alpha=0.7)
        plt.axvline(np.mean(all_similarities['incorrect']), color='orange', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'use_similarity_distributions_{data_name}_{api}_{metadata_setting}.png', dpi=300)
        
        print(f"Saved plots to use_similarity_{data_name}_{api}_{metadata_setting}.png and use_similarity_distributions_{data_name}_{api}_{metadata_setting}.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'gemini'
    metadata_setting = "with_user_metadata"
    
    # Calculate and visualize semantic similarities
    similarity_metrics = calculate_semantic_similarity(data_name, api, metadata_setting)
    
    # Save metrics to JSON file
    with open(f'use_similarity_metrics_{data_name}_{api}_{metadata_setting}.json', 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        serializable_metrics = json.loads(json.dumps(similarity_metrics, default=lambda x: float(x) if isinstance(x, np.number) else x))
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Saved semantic similarity metrics to use_similarity_metrics_{data_name}_{api}_{metadata_setting}.json")