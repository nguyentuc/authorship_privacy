import os
import json
import pandas as pd
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to compute Jaccard Similarity
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1 & set2) / len(set1 | set2)


def calculate_tfidf():
    data_name='speech'
    api = 'deepseek'
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/without_user_metadata/'
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    print(f"Dataset structure: {dataset}")

    # load all the text need to compute

    speakers = ['obama', 'bush', 'trump']
    avg_cosine_similarity = []
    for speaker in speakers:
        print(f"Working on {speaker}")
        # original text
        # sample text that has bigger than 50 words
        author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        

        # obfuscation from correct attribute
        df = pd.read_csv(root_path+'obfuscation_from_correct_attribute/'+speaker+'.csv')
        obfuscation_correct = []
        for index, row in df.iterrows():
            obfuscation_correct.append(row['Obfuscation'])
    
        # from incorrect
        df = pd.read_csv(root_path+'obfuscation_from_incorrect_attribute/'+speaker+'.csv')
        obfuscation_incorrect = []
        for index, row in df.iterrows():
            obfuscation_incorrect.append(row['Obfuscation'])

        # Combine both corpora for consistent TF-IDF vectorization
        all_texts = original_text + obfuscation_correct + obfuscation_incorrect

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split back into original and paraphrased matrices
        original_vectors = tfidf_matrix[:len(original_text)]
        obfuscation_correct_vectors = tfidf_matrix[len(original_text):2*len(original_text)]
        obfuscation_incorrect_vectors = tfidf_matrix[2*len(original_text):]
        

        # Compute metric between original and obfuscation correct
        similarity_scores = cosine_similarity(original_vectors, obfuscation_correct_vectors)
        jaccard_scores = [jaccard_similarity(original_text[i], obfuscation_correct[i]) for i in range(len(original_text))]
        mean_cosine_similarity1 = np.mean(similarity_scores)
        mean_jaccard_similarity = np.mean(jaccard_scores)
        # print("Mean Cosine Original - Obfucation Correct:", mean_cosine_similarity)
        # print("Mean Jaccard Original - Obfuscation Correct:", mean_jaccard_similarity)

        # Compute metric between original and obfuscation incorrect
        similarity_scores = cosine_similarity(original_vectors, obfuscation_incorrect_vectors)
        jaccard_scores = [jaccard_similarity(original_text[i], obfuscation_incorrect[i]) for i in range(len(original_text))]
        mean_cosine_similarity2 = np.mean(similarity_scores)
        mean_jaccard_similarity = np.mean(jaccard_scores)
        # print("Mean Cosine Original - Obfucation InCorrect:", mean_cosine_similarity)
        # print("Mean Jaccard Original - Obfucation InCorrect:", mean_jaccard_similarity)
        avg_cosine_similarity.append((mean_cosine_similarity1+mean_cosine_similarity2)/2)
    print("Avg cosine sim:", np.mean(avg_cosine_similarity))


calculate_tfidf()
