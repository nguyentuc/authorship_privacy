import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gensim
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_from_disk

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def analyze_topics_with_lda():
    """
    Analyzes speech data using Latent Dirichlet Allocation (LDA) to extract topics
    from original texts, obfuscated texts, and mimicked texts.
    """
    # Configuration
    root_path = '/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/without_user_metadata/'
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    speakers = ['obama', 'bush', 'trump']
    num_topics = 2
    num_words_per_topic = 10
    
    # Load the dataset
    print("Loading speech dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Prepare stopwords
    stop_words = set(stopwords.words('english'))
    additional_stop_words = [
        'and', 'also', 'the', 'to', 'we', '.', ',', "'s", '--', "n't", 
        "'ve", "''", 'us', ';', "'m", 'must', 'like', 'every', "'", "'re"
    ]
    stop_words.update(additional_stop_words)
    
    # Process each speaker
    for speaker in speakers:
        print(f"\n{'='*80}\nAnalyzing topics for {speaker.upper()}\n{'='*80}")
        
        # Load text datasets
        text_data = load_speaker_data(dataset, root_path, speaker, stop_words)
        
        # Process each text type with LDA
        for text_type, texts in text_data.items():
            print(f"\n{'-'*50}\n{text_type}:\n{'-'*50}")
            
            # Prepare the texts
            tokenized_texts = [word_tokenize(doc.lower()) for doc in texts]
            filtered_texts = [
                [word for word in doc if word not in stop_words]
                for doc in tokenized_texts
            ]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(filtered_texts)
            corpus = [dictionary.doc2bow(text) for text in filtered_texts]
            
            # Skip if corpus is empty
            if not corpus:
                print(f"Skipping {text_type} - no data available")
                continue
                
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=20,
                random_state=42
            )
            
            # Print topics
            topics = lda_model.print_topics(num_words=num_words_per_topic)
            for i, topic in enumerate(topics):
                print(f"Topic {i+1}: {topic}")
            
            # Calculate coherence score
            try:
                coherence_model = gensim.models.CoherenceModel(
                    model=lda_model,
                    texts=filtered_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                print(f"Coherence Score: {coherence_score:.4f}")
            except Exception as e:
                print(f"Could not calculate coherence score: {e}")
            
            # Get document topic distributions
            doc_topics = [lda_model.get_document_topics(bow) for bow in corpus[:5]]
            print("\nSample document topic distributions (first 5 documents):")
            for i, topics_dist in enumerate(doc_topics[:5]):
                topics_dist = sorted(topics_dist, key=lambda x: x[1], reverse=True)
                print(f"  Document {i+1}: {topics_dist}")


def load_speaker_data(dataset, root_path, speaker, stop_words):
    """
    Loads and prepares all data types for a given speaker.
    
    Args:
        dataset: The loaded speech dataset
        root_path: Root path to the transformed text files
        speaker: The speaker to analyze
        stop_words: Set of stopwords to filter
        
    Returns:
        Dictionary with text data organized by type
    """
    # Get original texts
    author_dataset = dataset.filter(
        lambda example: example["style"] == speaker and len(example["text"].split()) > 50
    )['train']
    author_dataset = author_dataset.shuffle(seed=2024)
    author_dataset = author_dataset.shuffle(seed=2025)
    author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
    original_text = [example['text'] for example in author_dataset]
    
    # Define paths for transformed text types
    file_paths = {
        'obfuscation': os.path.join(root_path, 'obfuscation', f'{speaker}.csv'),
        'mimicking_from_original': os.path.join(root_path, 'mimicking_from_original', f'{speaker}.csv'),
        'mimicking_from_obfuscation': os.path.join(root_path, 'mimicking_from_obfuscation', f'{speaker}.csv')
    }
    
    # Load transformed texts
    text_data = {'Original': original_text}
    
    for text_type, file_path in file_paths.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            column_name = 'Obfuscation' if 'obfuscation' in text_type else 'Mimicking'
            if column_name in df.columns:
                text_data[text_type.replace('_', ' ').title()] = df[column_name].tolist()
            else:
                print(f"Warning: Column '{column_name}' not found in {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return text_data


if __name__ == "__main__":
    analyze_topics_with_lda()