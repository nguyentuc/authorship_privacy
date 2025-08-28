import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def jaccard_similarity(text1, text2):
    """
    Compute the Jaccard similarity between two texts.
    
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


def calculate_text_similarities(data_name, api, metadata_setting="without_user_metadata"):
    """
    Calculate TF-IDF cosine similarity and Jaccard similarity between original and obfuscated texts.
    
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
    similarity_metrics = defaultdict(dict)
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nAnalyzing text similarities for {speaker}")
        
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
        
        # Combine all texts for consistent TF-IDF vectorization
        all_texts = original_text + obfuscation_correct + obfuscation_incorrect
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Split back into separate matrices
        original_vectors = tfidf_matrix[:min_length]
        obfuscation_correct_vectors = tfidf_matrix[min_length:2*min_length]
        obfuscation_incorrect_vectors = tfidf_matrix[2*min_length:3*min_length]
        
        # Calculate similarity metrics for original vs obfuscation correct
        cosine_sim_correct = cosine_similarity(original_vectors, obfuscation_correct_vectors)
        # Extract diagonal elements (pairwise comparisons)
        cosine_diag_correct = np.diag(cosine_sim_correct)
        mean_cosine_correct = np.mean(cosine_diag_correct)
        
        jaccard_scores_correct = [
            jaccard_similarity(original_text[i], obfuscation_correct[i]) 
            for i in range(min_length)
        ]
        mean_jaccard_correct = np.mean(jaccard_scores_correct)
        
        # Calculate similarity metrics for original vs obfuscation incorrect
        cosine_sim_incorrect = cosine_similarity(original_vectors, obfuscation_incorrect_vectors)
        cosine_diag_incorrect = np.diag(cosine_sim_incorrect)
        mean_cosine_incorrect = np.mean(cosine_diag_incorrect)
        
        jaccard_scores_incorrect = [
            jaccard_similarity(original_text[i], obfuscation_incorrect[i]) 
            for i in range(min_length)
        ]
        mean_jaccard_incorrect = np.mean(jaccard_scores_incorrect)
        
        # Store results
        similarity_metrics[speaker]['cosine_correct'] = mean_cosine_correct
        similarity_metrics[speaker]['cosine_incorrect'] = mean_cosine_incorrect
        similarity_metrics[speaker]['jaccard_correct'] = mean_jaccard_correct
        similarity_metrics[speaker]['jaccard_incorrect'] = mean_jaccard_incorrect
        similarity_metrics[speaker]['cosine_diag_correct'] = cosine_diag_correct.tolist()
        similarity_metrics[speaker]['cosine_diag_incorrect'] = cosine_diag_incorrect.tolist()
        similarity_metrics[speaker]['jaccard_scores_correct'] = jaccard_scores_correct
        similarity_metrics[speaker]['jaccard_scores_incorrect'] = jaccard_scores_incorrect
        
        # Calculate averages
        avg_cosine = (mean_cosine_correct + mean_cosine_incorrect) / 2
        avg_jaccard = (mean_jaccard_correct + mean_jaccard_incorrect) / 2
        
        print(f"  Cosine similarity - Original vs Correct: {mean_cosine_correct:.4f}")
        print(f"  Cosine similarity - Original vs Incorrect: {mean_cosine_incorrect:.4f}")
        print(f"  Average cosine similarity: {avg_cosine:.4f}")
        print(f"  Jaccard similarity - Original vs Correct: {mean_jaccard_correct:.4f}")
        print(f"  Jaccard similarity - Original vs Incorrect: {mean_jaccard_incorrect:.4f}")
        print(f"  Average Jaccard similarity: {avg_jaccard:.4f}")
    
    # Calculate overall averages
    avg_cosine_values = [
        (speaker_metrics['cosine_correct'] + speaker_metrics['cosine_incorrect']) / 2
        for speaker, speaker_metrics in similarity_metrics.items()
    ]
    
    avg_jaccard_values = [
        (speaker_metrics['jaccard_correct'] + speaker_metrics['jaccard_incorrect']) / 2
        for speaker, speaker_metrics in similarity_metrics.items()
    ]
    
    overall_avg_cosine = np.mean(avg_cosine_values)
    overall_avg_jaccard = np.mean(avg_jaccard_values)
    
    print(f"\nOverall average cosine similarity: {overall_avg_cosine:.4f}")
    print(f"Overall average Jaccard similarity: {overall_avg_jaccard:.4f}")
    
    # Create visualizations
    create_similarity_plots(similarity_metrics, data_name, api, metadata_setting)
    
    return similarity_metrics


def create_similarity_plots(similarity_metrics, data_name, api, metadata_setting):
    """
    Create visualizations for similarity metrics.
    
    Args:
        similarity_metrics: Dictionary containing similarity metrics
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: Setting used
    """
    try:
        # Convert to DataFrame for easier plotting
        plot_data = []
        for speaker, metrics in similarity_metrics.items():
            plot_data.append({
                'Speaker': speaker.capitalize(),
                'Metric': 'Cosine Similarity',
                'Type': 'Correct Attribution',
                'Value': metrics['cosine_correct']
            })
            plot_data.append({
                'Speaker': speaker.capitalize(),
                'Metric': 'Cosine Similarity',
                'Type': 'Incorrect Attribution',
                'Value': metrics['cosine_incorrect']
            })
            plot_data.append({
                'Speaker': speaker.capitalize(),
                'Metric': 'Jaccard Similarity',
                'Type': 'Correct Attribution',
                'Value': metrics['jaccard_correct']
            })
            plot_data.append({
                'Speaker': speaker.capitalize(),
                'Metric': 'Jaccard Similarity',
                'Type': 'Incorrect Attribution',
                'Value': metrics['jaccard_incorrect']
            })
        
        df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Speaker', y='Value', hue='Type', data=df[df['Metric'] == 'Cosine Similarity'])
        ax.set_title(f'Cosine Similarity: Original vs Obfuscated Texts ({api})', fontsize=14)
        ax.set_ylabel('Cosine Similarity')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'cosine_similarity_{data_name}_{api}_{metadata_setting}.png', dpi=300)
        
        # Create distribution plots
        plt.figure(figsize=(15, 10))
        
        for i, speaker in enumerate(similarity_metrics.keys()):
            plt.subplot(2, len(similarity_metrics), i+1)
            sns.histplot(similarity_metrics[speaker]['cosine_diag_correct'], 
                         kde=True, bins=15, alpha=0.6, label='Correct Attribution')
            sns.histplot(similarity_metrics[speaker]['cosine_diag_incorrect'], 
                         kde=True, bins=15, alpha=0.6, label='Incorrect Attribution')
            plt.title(f'{speaker.capitalize()}: Cosine Similarity Distribution')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.subplot(2, len(similarity_metrics), i+len(similarity_metrics)+1)
            sns.histplot(similarity_metrics[speaker]['jaccard_scores_correct'], 
                         kde=True, bins=15, alpha=0.6, label='Correct Attribution')
            sns.histplot(similarity_metrics[speaker]['jaccard_scores_incorrect'], 
                         kde=True, bins=15, alpha=0.6, label='Incorrect Attribution')
            plt.title(f'{speaker.capitalize()}: Jaccard Similarity Distribution')
            plt.xlabel('Jaccard Similarity')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'similarity_distributions_{data_name}_{api}_{metadata_setting}.png', dpi=300)
        
        print(f"Saved plots to cosine_similarity_{data_name}_{api}_{metadata_setting}.png and similarity_distributions_{data_name}_{api}_{metadata_setting}.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'deepseek'
    metadata_setting = "without_user_metadata"
    
    # Calculate and visualize text similarities
    similarity_metrics = calculate_text_similarities(data_name, api, metadata_setting)
    
    # Save metrics to JSON file
    with open(f'similarity_metrics_{data_name}_{api}_{metadata_setting}.json', 'w') as f:
        json.dump(similarity_metrics, f, indent=4)
    
    print(f"Saved similarity metrics to similarity_metrics_{data_name}_{api}_{metadata_setting}.json")