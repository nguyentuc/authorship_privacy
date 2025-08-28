import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_from_disk
from tqdm import tqdm

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def analyze_topics_with_lda(data_name="speech", api="deepseek", metadata_setting="without_user_metadata", num_topics=2, num_words=10):
    """
    Analyze topics in original and obfuscated texts using Latent Dirichlet Allocation.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        metadata_setting: Setting to use ("with_user_metadata" or "without_user_metadata")
        num_topics: Number of topics to extract
        num_words: Number of words to display per topic
        
    Returns:
        Dictionary containing extracted topics for each speaker and text type
    """
    # Set paths
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_setting}/'
    output_dir = f'topic_analysis_results/{data_name}/{api}/{metadata_setting}/'
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Prepare stopwords
    stop_words = set(stopwords.words('english'))
    additional_stop_words = [
        'and', 'also', 'the', 'to', 'we', '.', ',', "'s", '--', "n't", 
        "'ve", "''", 'us', ';', "'m", 'must', 'like', 'every', "'", "'re",
        'a', 'of', 'in', 'is', 'it', 'that', 'this', 'for', 'be', 'are',
        'on', 'with', 'as', 'by', 'at', 'from', 'an', 'not', 'have', 'has',
        'was', 'were', 'would', 'could', 'should', 'can', 'will', 'do', 'does'
    ]
    stop_words.update(additional_stop_words)
    
    # Dictionary to store topic analysis results
    topic_results = {}
    
    # Process each speaker
    for speaker in speakers:
        print(f"\n{'='*60}\nAnalyzing topics for {speaker.upper()}\n{'='*60}")
        topic_results[speaker] = {}
        
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
        
        # Ensure all text lists have a reasonable length
        min_length = min(len(original_text), len(obfuscation_correct), len(obfuscation_incorrect))
        if min_length < 10:
            print(f"Warning: Not enough samples for {speaker}, skipping")
            continue
            
        original_text = original_text[:min_length]
        obfuscation_correct = obfuscation_correct[:min_length]
        obfuscation_incorrect = obfuscation_incorrect[:min_length]
        
        print(f"Processing {min_length} text samples for each category")
        
        # Process each text type with LDA
        text_types = {
            'Original': original_text,
            'Obfuscation_Correct': obfuscation_correct,
            'Obfuscation_Incorrect': obfuscation_incorrect
        }
        
        for text_type, texts in text_types.items():
            print(f"\n{'-'*50}\nAnalyzing topics for {text_type}\n{'-'*50}")
            
            # Tokenize and clean texts
            tokenized_texts = []
            for doc in tqdm(texts, desc="Tokenizing texts"):
                tokens = word_tokenize(doc.lower())
                filtered_tokens = [word for word in tokens if word not in stop_words]
                tokenized_texts.append(filtered_tokens)
            
            # Create dictionary
            dictionary = corpora.Dictionary(tokenized_texts)
            
            # Filter extremes (remove very rare and very common words)
            dictionary.filter_extremes(no_below=2, no_above=0.9)
            
            # Create corpus
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Skip if corpus is empty
            if not corpus:
                print(f"Skipping {text_type} - no data available after filtering")
                continue
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=20,
                random_state=42
            )
            
            # Get topics
            topics = lda_model.print_topics(num_words=num_words)
            
            # Store topics
            topic_results[speaker][text_type] = {
                'topics': [topic for topic in topics],
                'top_words': {}
            }
            
            # Extract and store top words for each topic
            for topic_id, topic_words in topics:
                # Parse the topic string to extract words and weights
                words_with_weights = []
                for word_weight in topic_words.split(" + "):
                    parts = word_weight.split("*")
                    if len(parts) == 2:
                        weight = float(parts[0])
                        word = parts[1].strip('"')
                        words_with_weights.append((word, weight))
                
                topic_results[speaker][text_type]['top_words'][f'Topic_{topic_id}'] = words_with_weights
            
            # Print topics
            for topic_id, topic_words in topics:
                print(f"Topic {topic_id}: {topic_words}")
            
            # Calculate coherence score
            try:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                topic_results[speaker][text_type]['coherence_score'] = coherence_score
                print(f"Coherence Score: {coherence_score:.4f}")
            except Exception as e:
                print(f"Could not calculate coherence score: {e}")
                topic_results[speaker][text_type]['coherence_score'] = None
            
            # Calculate document-topic distributions
            doc_topic_dist = []
            for doc in corpus:
                topic_dist = lda_model.get_document_topics(doc)
                # Convert to a full distribution over all topics
                dist = [0] * num_topics
                for topic_id, prob in topic_dist:
                    dist[topic_id] = prob
                doc_topic_dist.append(dist)
            
            # Calculate average topic distribution
            avg_topic_dist = np.mean(doc_topic_dist, axis=0) if doc_topic_dist else []
            topic_results[speaker][text_type]['avg_topic_distribution'] = avg_topic_dist.tolist()
            
            print(f"Average topic distribution: {avg_topic_dist}")
    
    # Create visualizations
    create_topic_visualizations(topic_results, data_name, api, metadata_setting, num_topics)
    
    # Save results
    with open(os.path.join(output_dir, "topic_analysis_results.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(json.dumps(topic_results, default=lambda x: float(x) if isinstance(x, np.number) else x))
        json.dump(serializable_results, f, indent=4)
    
    return topic_results


def create_topic_visualizations(topic_results, data_name, api, metadata_setting, num_topics):
    """
    Create visualizations for topic analysis results.
    
    Args:
        topic_results: Dictionary containing topic analysis results
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: Setting used
        num_topics: Number of topics extracted
    """
    if not topic_results:
        print("No data available for visualization")
        return
        
    try:
        output_dir = f'topic_analysis_results/{data_name}/{api}/{metadata_setting}/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for coherence score comparison
        speakers = list(topic_results.keys())
        text_types = ['Original', 'Obfuscation_Correct', 'Obfuscation_Incorrect']
        
        coherence_data = []
        for speaker in speakers:
            for text_type in text_types:
                if text_type in topic_results[speaker] and 'coherence_score' in topic_results[speaker][text_type]:
                    coherence_score = topic_results[speaker][text_type]['coherence_score']
                    if coherence_score is not None:
                        coherence_data.append({
                            'Speaker': speaker.capitalize(),
                            'Text Type': text_type.replace('_', ' '),
                            'Coherence Score': coherence_score
                        })
        
        if coherence_data:
            # Create coherence score comparison plot
            plt.figure(figsize=(12, 6))
            
            df = pd.DataFrame(coherence_data)
            ax = sns.barplot(x='Speaker', y='Coherence Score', hue='Text Type', data=df)
            
            plt.title('Topic Coherence Comparison', fontsize=14)
            plt.ylabel('Coherence Score (c_v)')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "coherence_scores.png"), dpi=300)
            
            # Create topic distribution comparison plots for each speaker
            for speaker in speakers:
                plt.figure(figsize=(10, 6))
                
                topic_distributions = []
                for text_type in text_types:
                    if text_type in topic_results[speaker] and 'avg_topic_distribution' in topic_results[speaker][text_type]:
                        dist = topic_results[speaker][text_type]['avg_topic_distribution']
                        if dist:
                            topic_distributions.append(dist)
                
                if len(topic_distributions) == len(text_types):
                    x = np.arange(num_topics)
                    width = 0.25
                    
                    for i, (text_type, dist) in enumerate(zip(text_types, topic_distributions)):
                        plt.bar(x + (i - 1) * width, dist, width, label=text_type.replace('_', ' '))
                    
                    plt.xlabel('Topic ID')
                    plt.ylabel('Average Topic Weight')
                    plt.title(f'Topic Distribution Comparison: {speaker.capitalize()}')
                    plt.xticks(x, [f'Topic {i}' for i in range(num_topics)])
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{speaker}_topic_distribution.png"), dpi=300)
                
                # Create word cloud visualization for top words
                plt.figure(figsize=(15, 10))
                
                for i, text_type in enumerate(text_types):
                    if text_type in topic_results[speaker] and 'top_words' in topic_results[speaker][text_type]:
                        top_words = topic_results[speaker][text_type]['top_words']
                        
                        for j, topic_id in enumerate(sorted(top_words.keys())):
                            plt.subplot(len(text_types), num_topics, i * num_topics + j + 1)
                            
                            words_weights = top_words[topic_id]
                            if words_weights:
                                words = [word for word, _ in words_weights]
                                weights = [weight for _, weight in words_weights]
                                
                                # Create horizontal bar chart for top words
                                y_pos = np.arange(len(words))
                                plt.barh(y_pos, weights, align='center')
                                plt.yticks(y_pos, words)
                                plt.title(f'{text_type.replace("_", " ")}: {topic_id}')
                                plt.tight_layout()
                
                plt.suptitle(f'Top Words by Topic: {speaker.capitalize()}', fontsize=16, y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{speaker}_top_words.png"), dpi=300, bbox_inches='tight')
        
        print(f"Saved visualizations to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")


# Main execution
if __name__ == "__main__":
    import seaborn as sns
    
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")
    
    # Run topic analysis
    data_name = "speech"
    api = "deepseek"
    metadata_setting = "without_user_metadata"
    
    topic_results = analyze_topics_with_lda(
        data_name=data_name,
        api=api,
        metadata_setting=metadata_setting,
        num_topics=2,
        num_words=10
    )