import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_from_disk
from tqdm import tqdm

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
    output_dir = f'semantic_similarity_results/{data_name}/{api}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the Universal Sentence Encoder model
    print("Loading Universal Sentence Encoder...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Load dataset
    print(f"Loading {data_name} dataset...")
    try:
        if data_name == 'speech':
            dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
            print(f"Dataset structure: {dataset}")
            speakers = ['obama', 'bush', 'trump']
        else:
            # For Quora dataset, get author IDs from the directory
            profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
            speakers = [f.split('.')[0] for f in os.listdir(profile_dir) if f.endswith('.txt')]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}
    
    # Dictionary to store similarity results
    similarity_results = {}
    overall_correct_similarities = []
    overall_incorrect_similarities = []
    
    # Process each speaker/author
    for speaker in speakers:
        print(f"\n{'='*40}\nWorking on {speaker}\n{'='*40}")
        
        # Get original texts
        if data_name == 'speech':
            author_dataset = dataset.filter(
                lambda example: example["style"] == speaker and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            original_text = [example['text'] for example in author_dataset]
        else:
            # For Quora dataset, load from CSV
            writing_file = f'/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/{speaker}.csv'
            if not os.path.exists(writing_file):
                print(f"Warning: File not found - {writing_file}")
                continue
                
            df_original = pd.read_csv(writing_file)
            df_original = df_original.sample(frac=0.2, random_state=42)
            original_text = [row['Question'] + ' ' + row['Answer'] for _, row in df_original.iterrows()]
        
        # Load obfuscation from correct attribute
        correct_path = os.path.join(root_path, 'obfuscation_from_correct_attribute', f'{speaker}.csv')
        if not os.path.exists(correct_path):
            print(f"Warning: File not found - {correct_path}")
            continue
            
        df_correct = pd.read_csv(correct_path)
        obfuscation_correct = df_correct['Obfuscation'].tolist()
        
        # Load obfuscation from incorrect attribute
        incorrect_path = os.path.join(root_path, 'obfuscation_from_incorrect_attribute', f'{speaker}.csv')
        if not os.path.exists(incorrect_path):
            print(f"Warning: File not found - {incorrect_path}")
            continue
            
        df_incorrect = pd.read_csv(incorrect_path)
        obfuscation_incorrect = df_incorrect['Obfuscation'].tolist()
        
        # Ensure all text lists have the same length
        min_length = min(len(original_text), len(obfuscation_correct), len(obfuscation_incorrect))
        original_text = original_text[:min_length]
        obfuscation_correct = obfuscation_correct[:min_length]
        obfuscation_incorrect = obfuscation_incorrect[:min_length]
        
        print(f"Processing {min_length} text samples for each category")
        
        if min_length == 0:
            print("No samples to analyze, skipping")
            continue
        
        # Compute embeddings using Universal Sentence Encoder
        print("Computing embeddings...")
        
        # Process in batches to avoid memory issues with large datasets
        batch_size = 32
        all_original_embeddings = []
        all_correct_embeddings = []
        all_incorrect_embeddings = []
        
        # Process original texts
        for i in tqdm(range(0, min_length, batch_size), desc="Embedding original texts"):
            batch = original_text[i:min(i+batch_size, min_length)]
            embeddings = model(batch)
            all_original_embeddings.append(embeddings)
        
        # Process obfuscation from correct attribute
        for i in tqdm(range(0, min_length, batch_size), desc="Embedding correct obfuscation"):
            batch = obfuscation_correct[i:min(i+batch_size, min_length)]
            embeddings = model(batch)
            all_correct_embeddings.append(embeddings)
        
        # Process obfuscation from incorrect attribute
        for i in tqdm(range(0, min_length, batch_size), desc="Embedding incorrect obfuscation"):
            batch = obfuscation_incorrect[i:min(i+batch_size, min_length)]
            embeddings = model(batch)
            all_incorrect_embeddings.append(embeddings)
        
        # Concatenate embeddings
        original_embeddings = tf.concat(all_original_embeddings, axis=0)
        obfuscation_correct_embeddings = tf.concat(all_correct_embeddings, axis=0)
        obfuscation_incorrect_embeddings = tf.concat(all_incorrect_embeddings, axis=0)
        
        # Normalize embeddings for cosine similarity
        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_obfuscation_correct_embeddings = tf.nn.l2_normalize(obfuscation_correct_embeddings, axis=1)
        norm_obfuscation_incorrect_embeddings = tf.nn.l2_normalize(obfuscation_incorrect_embeddings, axis=1)
        
        # Compute cosine similarity matrix for correct obfuscation
        cosine_similarity_matrix_correct = tf.matmul(
            norm_original_embeddings, 
            norm_obfuscation_correct_embeddings, 
            transpose_b=True
        )
        
        # Extract diagonal elements (pairwise similarities)
        diagonal_similarities_correct = tf.linalg.diag_part(cosine_similarity_matrix_correct)
        mean_similarity_correct = tf.reduce_mean(diagonal_similarities_correct)
        
        # Compute cosine similarity matrix for incorrect obfuscation
        cosine_similarity_matrix_incorrect = tf.matmul(
            norm_original_embeddings, 
            norm_obfuscation_incorrect_embeddings, 
            transpose_b=True
        )
        
        diagonal_similarities_incorrect = tf.linalg.diag_part(cosine_similarity_matrix_incorrect)
        mean_similarity_incorrect = tf.reduce_mean(diagonal_similarities_incorrect)
        
        # Print results
        print(f"Semantic similarity (Original vs Correct Obfuscation): {mean_similarity_correct:.4f}")
        print(f"Semantic similarity (Original vs Incorrect Obfuscation): {mean_similarity_incorrect:.4f}")
        
        # Store results
        similarity_results[speaker] = {
            'use_correct': float(mean_similarity_correct.numpy()),
            'use_incorrect': float(mean_similarity_incorrect.numpy()),
            'use_correct_distribution': diagonal_similarities_correct.numpy().tolist(),
            'use_incorrect_distribution': diagonal_similarities_incorrect.numpy().tolist()
        }
        
        # Accumulate for overall statistics
        overall_correct_similarities.extend(diagonal_similarities_correct.numpy())
        overall_incorrect_similarities.extend(diagonal_similarities_incorrect.numpy())
        
        # Save speaker-specific results
        with open(os.path.join(output_dir, f"{speaker}_semantic_similarity.json"), "w") as f:
            json.dump(similarity_results[speaker], f, indent=4)
    
    # Calculate overall statistics
    if overall_correct_similarities and overall_incorrect_similarities:
        overall_mean_correct = np.mean(overall_correct_similarities)
        overall_mean_incorrect = np.mean(overall_incorrect_similarities)
        
        print(f"\nOverall semantic similarity statistics:")
        print(f"  Mean similarity (Original vs Correct Obfuscation): {overall_mean_correct:.4f}")
        print(f"  Mean similarity (Original vs Incorrect Obfuscation): {overall_mean_incorrect:.4f}")
        
        # Add overall statistics to results
        similarity_results['overall'] = {
            'mean_correct': float(overall_mean_correct),
            'mean_incorrect': float(overall_mean_incorrect),
            'correct_distribution': overall_correct_similarities,
            'incorrect_distribution': overall_incorrect_similarities
        }
    
    # Create visualizations
    create_similarity_visualizations(similarity_results, data_name, api, metadata_setting)
    
    # Save overall results
    with open(os.path.join(output_dir, f"all_semantic_similarity_{metadata_setting}.json"), "w") as f:
        json.dump(similarity_results, f, indent=4)
    
    return similarity_results


def create_similarity_visualizations(similarity_results, data_name, api, metadata_setting):
    """
    Create visualizations for semantic similarity results.
    
    Args:
        similarity_results: Dictionary containing similarity metrics
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: Setting used
    """
    if not similarity_results or 'overall' not in similarity_results:
        print("Insufficient data for visualization")
        return
        
    try:
        output_dir = f'semantic_similarity_results/{data_name}/{api}/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract speaker names (excluding 'overall')
        speakers = [speaker for speaker in similarity_results.keys() if speaker != 'overall']
        
        if not speakers:
            print("No speaker data available for visualization")
            return
        
        # Prepare data for bar chart
        correct_scores = [similarity_results[speaker]['use_correct'] for speaker in speakers]
        incorrect_scores = [similarity_results[speaker]['use_incorrect'] for speaker in speakers]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(speakers))
        width = 0.35
        
        plt.bar(x - width/2, correct_scores, width, label='Correct Attribution', color='skyblue')
        plt.bar(x + width/2, incorrect_scores, width, label='Incorrect Attribution', color='lightcoral')
        
        plt.xlabel('Speaker')
        plt.ylabel('Semantic Similarity (USE)')
        plt.title(f'Semantic Similarity: Original vs Obfuscated Texts ({data_name.capitalize()}, {api})')
        plt.xticks(x, [speaker.capitalize() for speaker in speakers])
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"semantic_similarity_bar_{metadata_setting}.png"), dpi=300)
        
        # Create distribution plots
        plt.figure(figsize=(15, 10))
        
        # Add distribution plots for each speaker
        for i, speaker in enumerate(speakers):
            plt.subplot(len(speakers) + 1, 1, i+1)
            
            correct_dist = similarity_results[speaker]['use_correct_distribution']
            incorrect_dist = similarity_results[speaker]['use_incorrect_distribution']
            
            sns.histplot(correct_dist, kde=True, bins=20, alpha=0.6, label='Correct Attribution', color='skyblue')
            sns.histplot(incorrect_dist, kde=True, bins=20, alpha=0.6, label='Incorrect Attribution', color='lightcoral')
            
            plt.title(f'Semantic Similarity Distribution: {speaker.capitalize()}')
            plt.xlabel('Semantic Similarity (USE)')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Add vertical lines for means
            plt.axvline(np.mean(correct_dist), color='blue', linestyle='--', alpha=0.7)
            plt.axvline(np.mean(incorrect_dist), color='red', linestyle='--', alpha=0.7)
        
        # Add overall distribution
        plt.subplot(len(speakers) + 1, 1, len(speakers) + 1)
        
        overall_correct = similarity_results['overall']['correct_distribution']
        overall_incorrect = similarity_results['overall']['incorrect_distribution']
        
        sns.histplot(overall_correct, kde=True, bins=20, alpha=0.6, label='Correct Attribution', color='skyblue')
        sns.histplot(overall_incorrect, kde=True, bins=20, alpha=0.6, label='Incorrect Attribution', color='lightcoral')
        
        plt.title('Overall Semantic Similarity Distribution')
        plt.xlabel('Semantic Similarity (USE)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Add vertical lines for means
        plt.axvline(np.mean(overall_correct), color='blue', linestyle='--', alpha=0.7)
        plt.axvline(np.mean(overall_incorrect), color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"semantic_similarity_dist_{metadata_setting}.png"), dpi=300)
        
        print(f"Saved visualizations to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'gemini'
    metadata_setting = "with_user_metadata"
    
    # Calculate semantic similarity
    similarity_results = calculate_semantic_similarity(data_name, api, metadata_setting)