import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from tqdm import tqdm

def jaccard_similarity(text1, text2):
    """
    Compute Jaccard similarity between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Jaccard similarity score (0-1)
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
        
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_text_similarity_metrics(data_name, api, metadata_setting="without_user_metadata"):
    """
    Calculate text similarity metrics (TF-IDF cosine similarity and Jaccard similarity)
    between original and obfuscated texts.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        metadata_setting: Setting to use ("with_user_metadata" or "without_user_metadata")
        
    Returns:
        Dictionary containing similarity metrics for each speaker
    """
    # Set paths
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_setting}/'
    
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
    
    # Prepare results containers
    similarity_results = {}
    cosine_similarity_values = []
    jaccard_similarity_values = []
    
    # Process each speaker/author
    for speaker in speakers:
        print(f"\n{'='*40}\nWorking on {speaker}\n{'='*40}")
        similarity_results[speaker] = {}
        
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
        
        print(f"Analyzing {min_length} text samples for each category")
        
        if min_length == 0:
            print("No samples to analyze, skipping")
            continue
        
        # Combine all texts for consistent TF-IDF vectorization
        all_texts = original_text + obfuscation_correct + obfuscation_incorrect
        
        # Compute TF-IDF vectors
        print("Computing TF-IDF vectors...")
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to improve performance
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Split back into separate matrices
        original_vectors = tfidf_matrix[:min_length]
        obfuscation_correct_vectors = tfidf_matrix[min_length:2*min_length]
        obfuscation_incorrect_vectors = tfidf_matrix[2*min_length:3*min_length]
        
        # Compute similarity metrics between original and obfuscation from correct attribute
        print("Computing similarity metrics for obfuscation from correct attribute...")
        
        # Cosine similarity
        similarity_matrix_correct = cosine_similarity(original_vectors, obfuscation_correct_vectors)
        cosine_diag_correct = np.diag(similarity_matrix_correct)  # Get pairwise similarities
        mean_cosine_correct = np.mean(cosine_diag_correct)
        
        # Jaccard similarity
        jaccard_correct = []
        for i in tqdm(range(min_length), desc="Jaccard similarity (correct)"):
            jaccard_correct.append(jaccard_similarity(original_text[i], obfuscation_correct[i]))
        mean_jaccard_correct = np.mean(jaccard_correct)
        
        # Compute similarity metrics between original and obfuscation from incorrect attribute
        print("Computing similarity metrics for obfuscation from incorrect attribute...")
        
        # Cosine similarity
        similarity_matrix_incorrect = cosine_similarity(original_vectors, obfuscation_incorrect_vectors)
        cosine_diag_incorrect = np.diag(similarity_matrix_incorrect)  # Get pairwise similarities
        mean_cosine_incorrect = np.mean(cosine_diag_incorrect)
        
        # Jaccard similarity
        jaccard_incorrect = []
        for i in tqdm(range(min_length), desc="Jaccard similarity (incorrect)"):
            jaccard_incorrect.append(jaccard_similarity(original_text[i], obfuscation_incorrect[i]))
        mean_jaccard_incorrect = np.mean(jaccard_incorrect)
        
        # Calculate average similarity metrics
        avg_cosine = (mean_cosine_correct + mean_cosine_incorrect) / 2
        avg_jaccard = (mean_jaccard_correct + mean_jaccard_incorrect) / 2
        
        # Store results
        similarity_results[speaker] = {
            'cosine_correct': mean_cosine_correct,
            'cosine_incorrect': mean_cosine_incorrect,
            'jaccard_correct': mean_jaccard_correct,
            'jaccard_incorrect': mean_jaccard_incorrect,
            'avg_cosine': avg_cosine,
            'avg_jaccard': avg_jaccard,
            'cosine_diag_correct': cosine_diag_correct.tolist(),
            'cosine_diag_incorrect': cosine_diag_incorrect.tolist(),
            'jaccard_correct': jaccard_correct,
            'jaccard_incorrect': jaccard_incorrect
        }
        
        # Accumulate for overall average
        cosine_similarity_values.append(avg_cosine)
        jaccard_similarity_values.append(avg_jaccard)
        
        # Print results
        print(f"\nSimilarity metrics for {speaker}:")
        print(f"  Cosine similarity (correct): {mean_cosine_correct:.4f}")
        print(f"  Cosine similarity (incorrect): {mean_cosine_incorrect:.4f}")
        print(f"  Average cosine similarity: {avg_cosine:.4f}")
        print(f"  Jaccard similarity (correct): {mean_jaccard_correct:.4f}")
        print(f"  Jaccard similarity (incorrect): {mean_jaccard_incorrect:.4f}")
        print(f"  Average Jaccard similarity: {avg_jaccard:.4f}")
    
    # Calculate overall averages
    if cosine_similarity_values:
        overall_avg_cosine = np.mean(cosine_similarity_values)
        overall_avg_jaccard = np.mean(jaccard_similarity_values)
        
        print(f"\nOverall average cosine similarity: {overall_avg_cosine:.4f}")
        print(f"Overall average Jaccard similarity: {overall_avg_jaccard:.4f}")
    
    # Create visualizations
    create_similarity_visualizations(similarity_results, data_name, api, metadata_setting)
    
    return similarity_results


def create_similarity_visualizations(similarity_results, data_name, api, metadata_setting):
    """
    Create visualizations for text similarity metrics.
    
    Args:
        similarity_results: Dictionary containing similarity metrics
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: Setting used
    """
    if not similarity_results:
        print("No data available for visualization")
        return
        
    try:
        # Prepare data for bar chart
        speakers = list(similarity_results.keys())
        cosine_correct = [similarity_results[s]['cosine_correct'] for s in speakers]
        cosine_incorrect = [similarity_results[s]['cosine_incorrect'] for s in speakers]
        jaccard_correct = [similarity_results[s]['jaccard_correct'] for s in speakers]
        jaccard_incorrect = [similarity_results[s]['jaccard_incorrect'] for s in speakers]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar width and positions
        width = 0.35
        x = np.arange(len(speakers))
        
        # Plot cosine similarity
        ax1.bar(x - width/2, cosine_correct, width, label='Correct Attribution', color='skyblue')
        ax1.bar(x + width/2, cosine_incorrect, width, label='Incorrect Attribution', color='lightcoral')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title('TF-IDF Cosine Similarity')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.capitalize() for s in speakers])
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot Jaccard similarity
        ax2.bar(x - width/2, jaccard_correct, width, label='Correct Attribution', color='skyblue')
        ax2.bar(x + width/2, jaccard_incorrect, width, label='Incorrect Attribution', color='lightcoral')
        ax2.set_ylabel('Jaccard Similarity')
        ax2.set_title('Jaccard Similarity')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.capitalize() for s in speakers])
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Text Similarity Metrics: {data_name.capitalize()} Dataset ({api})', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_file = f'text_similarity_{data_name}_{api}_{metadata_setting}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved similarity metrics visualization to {output_file}")
        
        # Create distribution plots
        plt.figure(figsize=(15, 10))
        
        for i, speaker in enumerate(speakers):
            # Get data
            cosine_correct_dist = similarity_results[speaker]['cosine_diag_correct']
            cosine_incorrect_dist = similarity_results[speaker]['cosine_diag_incorrect']
            
            # Plot distribution of cosine similarity values
            plt.subplot(len(speakers), 1, i+1)
            sns.histplot(cosine_correct_dist, kde=True, bins=20, alpha=0.6, 
                         label='Correct Attribution', color='skyblue')
            sns.histplot(cosine_incorrect_dist, kde=True, bins=20, alpha=0.6,
                         label='Incorrect Attribution', color='lightcoral')
            
            plt.title(f'Cosine Similarity Distribution: {speaker.capitalize()}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Add vertical lines for means
            plt.axvline(np.mean(cosine_correct_dist), color='blue', linestyle='--', alpha=0.7)
            plt.axvline(np.mean(cosine_incorrect_dist), color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save distribution plot
        output_file = f'similarity_distribution_{data_name}_{api}_{metadata_setting}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved similarity distribution visualization to {output_file}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'deepseek'
    metadata_setting = "without_user_metadata"
    
    # Calculate similarity metrics
    similarity_results = calculate_text_similarity_metrics(data_name, api, metadata_setting)
    
    # Save results to JSON
    output_file = f'text_similarity_metrics_{data_name}_{api}_{metadata_setting}.json'
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(json.dumps(similarity_results, default=lambda x: float(x) if isinstance(x, np.number) else x))
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved similarity metrics to {output_file}")