import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel
from collections import defaultdict

# Download necessary resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def analyze_topics_with_lda(metadata_setting="without_user_metadata"):
    """
    Analyze topics in original and obfuscated texts using Latent Dirichlet Allocation.
    
    Args:
        metadata_setting: Setting to use ("with_user_metadata" or "without_user_metadata")
    
    Returns:
        Dictionary containing extracted topics for each speaker and text type
    """
    # Set paths
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{metadata_setting}/'
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    
    # Prepare stopwords
    stop_words = set(stopwords.words('english'))
    additional_stop_words = [
        'and', 'also', 'the', 'to', 'we', '.', ',', "'s", '--', "n't", 
        "'ve", "''", 'us', ';', "'m", 'must', 'like', 'every', "'", "'re"
    ]
    stop_words.update(additional_stop_words)
    
    # Load dataset
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset structure: {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}
    
    # Define speakers
    speakers = ['obama', 'bush', 'trump']
    
    # Dictionary to store extracted topics
    topic_results = {}
    
    # Process each speaker
    for speaker in speakers:
        print(f"\n{'='*60}\nAnalyzing topics for {speaker.upper()}\n{'='*60}")
        topic_results[speaker] = {}
        
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
        
        # Process each text type with LDA
        text_types = {
            'Original': original_text,
            'Obfuscation_Correct': obfuscation_correct,
            'Obfuscation_Incorrect': obfuscation_incorrect
        }
        
        for text_type, texts in text_types.items():
            print(f"\n{'-'*50}\n{text_type}:\n{'-'*50}")
            
            # Tokenize and clean texts
            tokenized_texts = [word_tokenize(doc.lower()) for doc in texts]
            
            # Remove stopwords
            cleaned_texts = []
            for doc_tokens in tokenized_texts:
                cleaned_doc = [word for word in doc_tokens if word not in stop_words]
                cleaned_texts.append(cleaned_doc)
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(cleaned_texts)
            corpus = [dictionary.doc2bow(text) for text in cleaned_texts]
            
            # Skip if corpus is empty
            if not corpus:
                print(f"Skipping {text_type} - no data available")
                continue
            
            # Train LDA model
            num_topics = 2  # Number of topics to extract
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                passes=20,
                random_state=42
            )
            
            # Get topics
            topics = lda_model.print_topics(num_words=10)
            
            # Store topics
            topic_results[speaker][text_type] = {}
            for i, topic in enumerate(topics):
                print(f"Topic {i+1}: {topic}")
                topic_results[speaker][text_type][f'Topic_{i+1}'] = topic
            
            # Calculate coherence score
            try:
                coherence_model = gensim.models.CoherenceModel(
                    model=lda_model,
                    texts=cleaned_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                print(f"Coherence Score: {coherence_score:.4f}")
                topic_results[speaker][text_type]['Coherence_Score'] = coherence_score
            except Exception as e:
                print(f"Could not calculate coherence score: {e}")
            
            # Calculate topic distributions for documents
            doc_topics = []
            for bow in corpus:
                topic_dist = lda_model.get_document_topics(bow)
                doc_topics.append(sorted(topic_dist, key=lambda x: x[1], reverse=True))
            
            # Calculate average topic distribution
            topic_dist_avg = defaultdict(float)
            for doc_dist in doc_topics:
                for topic_id, prob in doc_dist:
                    topic_dist_avg[topic_id] += prob / len(doc_topics)
            
            # Store average topic distribution
            topic_results[speaker][text_type]['Avg_Topic_Distribution'] = dict(topic_dist_avg)
            print(f"Average topic distribution: {dict(topic_dist_avg)}")
    
    # Create visualizations
    create_topic_visualization(topic_results, metadata_setting)
    
    return topic_results


def create_topic_visualization(topic_results, metadata_setting):
    """
    Create visualizations for topic analysis.
    
    Args:
        topic_results: Dictionary containing extracted topics
        metadata_setting: Setting used
    """
    try:
        # Extract coherence scores
        coherence_data = []
        
        for speaker, speaker_data in topic_results.items():
            for text_type, type_data in speaker_data.items():
                if 'Coherence_Score' in type_data:
                    coherence_data.append({
                        'Speaker': speaker.capitalize(),
                        'Text Type': text_type.replace('_', ' '),
                        'Coherence Score': type_data['Coherence_Score']
                    })
        
        # Create coherence score comparison plot
        if coherence_data:
            plt.figure(figsize=(12, 6))
            
            df = pd.DataFrame(coherence_data)
            ax = sns.barplot(x='Speaker', y='Coherence Score', hue='Text Type', data=df)
            
            plt.title('Topic Coherence Comparison', fontsize=14)
            plt.ylabel('Coherence Score (c_v)')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'topic_coherence_{metadata_setting}.png', dpi=300)
            print(f"Saved coherence plot to topic_coherence_{metadata_setting}.png")
        
        # Extract top words from topics for visualization
        topic_words = defaultdict(dict)
        
        for speaker, speaker_data in topic_results.items():
            for text_type, type_data in speaker_data.items():
                topic_word_list = []
                
                for topic_key, topic_info in type_data.items():
                    if topic_key.startswith('Topic_'):
                        # Parse topic string to extract words and weights
                        # Format example: '0.123*"word1" + 0.456*"word2" + ...'
                        topic_str = topic_info[1]  # Extract the topic string part
                        words_with_weights = topic_str.split(' + ')
                        
                        for word_weight in words_with_weights:
                            parts = word_weight.split('*')
                            if len(parts) == 2:
                                weight = float(parts[0])
                                word = parts[1].strip('"')
                                topic_word_list.append((word, weight))
                
                topic_words[speaker][text_type] = topic_word_list
        
        # Create word cloud-like visualization for each speaker
        for speaker, speaker_data in topic_words.items():
            plt.figure(figsize=(15, 10))
            
            for i, (text_type, word_list) in enumerate(speaker_data.items(), 1):
                plt.subplot(1, len(speaker_data), i)
                
                # Prepare data for horizontal bar chart
                words = [word for word, _ in word_list[:20]]  # Top 20 words
                weights = [weight for _, weight in word_list[:20]]
                
                # Sort by weight
                sorted_indices = np.argsort(weights)[::-1]
                words = [words[i] for i in sorted_indices]
                weights = [weights[i] for i in sorted_indices]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(words))
                plt.barh(y_pos, weights, align='center', alpha=0.7)
                plt.yticks(y_pos, words)
                plt.xlabel('Weight')
                plt.title(f'{text_type.replace("_", " ")}', fontsize=12)
                plt.tight_layout()
            
            plt.suptitle(f'Top Words in Topics - {speaker.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            plt.savefig(f'topic_words_{speaker}_{metadata_setting}.png', dpi=300)
            print(f"Saved topic words plot to topic_words_{speaker}_{metadata_setting}.png")
        
    except Exception as e:
        print(f"Error creating topic visualizations: {e}")


# Main execution
if __name__ == "__main__":
    metadata_setting = "without_user_metadata"
    
    # Analyze topics
    topic_results = analyze_topics_with_lda(metadata_setting)
    
    # Save results to JSON file
    with open(f'topic_analysis_{metadata_setting}.json', 'w') as f:
        # Convert objects to serializable types
        serializable_results = json.loads(json.dumps(topic_results, default=lambda x: float(x) if isinstance(x, np.number) else str(x)))
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved topic analysis results to topic_analysis_{metadata_setting}.json")