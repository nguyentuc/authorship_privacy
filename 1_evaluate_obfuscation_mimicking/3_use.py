import os
import json
import pandas as pd
from datasets import load_from_disk
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline

def analyze_text_similarities():
    """
    Analyze the semantic similarities between original texts and various 
    transformations (obfuscation, mimicking) using Universal Sentence Encoder.
    """
    # Configuration
    dataset_name = 'speech'
    api = 'gemini'
    root_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
    
    # Load the dataset
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    print(f"Dataset structure: {dataset}")
    
    # Load the Universal Sentence Encoder model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Load the sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    
    # Define speakers to analyze
    speakers = ['obama', 'bush', 'trump']
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nAnalyzing {speaker}'s texts:")
        
        # Get original texts
        author_dataset = dataset.filter(
            lambda example: example["style"] == speaker and len(example["text"].split()) > 50
        )['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        
        # Load obfuscation text
        obfuscation_path = os.path.join(root_path, 'obfuscation', f'{speaker}.csv')
        if not os.path.exists(obfuscation_path):
            print(f"Warning: {obfuscation_path} not found")
            continue
            
        df_obfuscation = pd.read_csv(obfuscation_path)
        obfuscation_text = df_obfuscation['Obfuscation'].tolist()
        
        # Load text mimicking from original
        mimicking_orig_path = os.path.join(root_path, 'mimicking_from_original', f'{speaker}.csv')
        if not os.path.exists(mimicking_orig_path):
            print(f"Warning: {mimicking_orig_path} not found")
            continue
            
        df_mimicking_orig = pd.read_csv(mimicking_orig_path)
        mimicking_text_from_orig = df_mimicking_orig['Mimicking'].tolist()
        
        # Load text mimicking from obfuscation
        mimicking_obf_path = os.path.join(root_path, 'mimicking_from_obfuscation', f'{speaker}.csv')
        if not os.path.exists(mimicking_obf_path):
            print(f"Warning: {mimicking_obf_path} not found")
            continue
            
        df_mimicking_obf = pd.read_csv(mimicking_obf_path)
        mimicking_text_from_obf = df_mimicking_obf['Mimicking'].tolist()
        
        # Ensure all text lists have the same length (for diagonal comparison)
        min_length = min(len(original_text), len(obfuscation_text), 
                         len(mimicking_text_from_orig), len(mimicking_text_from_obf))
        
        if min_length == 0:
            print(f"No data available for {speaker}")
            continue
            
        original_text = original_text[:min_length]
        obfuscation_text = obfuscation_text[:min_length]
        mimicking_text_from_orig = mimicking_text_from_orig[:min_length]
        mimicking_text_from_obf = mimicking_text_from_obf[:min_length]
        
        print(f"Analyzing {min_length} texts for each category")
        
        # Compute embeddings with Universal Sentence Encoder
        original_embeddings = model(original_text)
        obfuscation_embeddings = model(obfuscation_text)
        mimicking_orig_embeddings = model(mimicking_text_from_orig)
        mimicking_obf_embeddings = model(mimicking_text_from_obf)
        
        # Compute similarity: Original vs Obfuscation
        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_obfuscation_embeddings = tf.nn.l2_normalize(obfuscation_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(
            norm_original_embeddings, 
            norm_obfuscation_embeddings, 
            transpose_b=True
        )
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Similarity - Original vs Obfuscation: {mean_similarity:.4f}")
        
        # Compute similarity: Original vs Mimicking from Original
        norm_mimicking_orig_embeddings = tf.nn.l2_normalize(mimicking_orig_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(
            norm_original_embeddings, 
            norm_mimicking_orig_embeddings, 
            transpose_b=True
        )
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Similarity - Original vs Mimicking from Original: {mean_similarity:.4f}")
        
        # Compute similarity: Original vs Mimicking from Obfuscation
        norm_mimicking_obf_embeddings = tf.nn.l2_normalize(mimicking_obf_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(
            norm_original_embeddings, 
            norm_mimicking_obf_embeddings, 
            transpose_b=True
        )
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Similarity - Original vs Mimicking from Obfuscation: {mean_similarity:.4f}")
        
        # Additional similarity: Obfuscation vs Mimicking from Obfuscation
        cosine_similarity_matrix = tf.matmul(
            norm_obfuscation_embeddings, 
            norm_mimicking_obf_embeddings, 
            transpose_b=True
        )
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Similarity - Obfuscation vs Mimicking from Obfuscation: {mean_similarity:.4f}")


if __name__ == "__main__":
    analyze_text_similarities()