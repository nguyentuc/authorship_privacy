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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import pandas as pd
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
import numpy as np




def compute_vocabulary_features(api, dataset_name, with_without):

    if dataset_name == "speech":
        root_path = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/{with_without}/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        print(f"Dataset structure: {dataset}")

        # load all the text need to compute

        speakers = ['obama', 'bush', 'trump']
        cosine_similarity_logs = {}
        for speaker in speakers:
            cosine_similarity_logs[speaker] = {}

            # original
            author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            original_text = [example['text'] for example in author_dataset]
            
            # text mimicking from correct 
            df = pd.read_csv(root_path+'mimicking_from_correct_attribute/'+speaker+'.csv')
            text_from_correct = []
            for index, row in df.iterrows():
                text_from_correct.append(row['Mimicking'].replace('\n', ' '))
            
            # text mimicking from incorrect
            df = pd.read_csv(root_path+'mimicking_from_incorrect_attribute/'+speaker+'.csv')
            mimicking_from_incorrect = []
            for index, row in df.iterrows():
                mimicking_from_incorrect.append(row['Mimicking'].replace("\n", ''))

            
            # Combine both corpora for consistent TF-IDF vectorization
            all_texts = original_text + text_from_correct + mimicking_from_incorrect

            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Split back into original and paraphrased matrices
            original_vectors = tfidf_matrix[:len(original_text)]
            text_from_correct_vectors = tfidf_matrix[len(original_text):2*len(original_text)]
            mimicking_from_incorrect_vectors = tfidf_matrix[2*len(original_text):]
            
            # Compute metric between original and obfuscation from correct
            similarity_scores = cosine_similarity(original_vectors, text_from_correct_vectors) # compute element-wise on the row of the matrix
            original_mimicking_from_correct = np.mean(similarity_scores) 

            # Compute metric between original and obfuscation from incorrect
            similarity_scores = cosine_similarity(original_vectors, mimicking_from_incorrect_vectors)
            original_mimicking_from_incorrect = np.mean(similarity_scores)
            cosine_similarity_logs[speaker] ={"original_mimicking_from_correct": original_mimicking_from_correct, "original_mimicking_from_incorrect": original_mimicking_from_incorrect}
        return cosine_similarity_logs


    else: # processing for quora
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        cosine_similarity_logs = {}
        for filename in os.listdir(root_path):
            speaker = filename.split('.')[0]
            cosine_similarity_logs[speaker] = {}

            # original text
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)
            original_text = [example['Question']+' '+example['Answer'] for idx, example in author_dataset.iterrows() ]
        
            # text mimicking from correct
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/{with_without}/mimicking_from_correct_attribute/'+speaker+'.csv')
            mimicking_text_from_correct = []
            for index, row in df.iterrows():
                mimicking_text_from_correct.append(row['Mimicking'].replace("\n", " "))
            
            # text obfuscation from incorrect
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/{with_without}/mimicking_from_incorrect_attribute/'+speaker+'.csv')
            mimicking_text_from_incorrect = []
            for index, row in df.iterrows():
                mimicking_text_from_incorrect.append(row['Mimicking'].replace('\n', " "))  
            
            # Combine both corpora for consistent TF-IDF vectorization
            all_texts = original_text +  mimicking_text_from_correct + mimicking_text_from_incorrect

            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Split back into original and paraphrased matrices
            original_vectors = tfidf_matrix[:len(original_text)]
            mimicking_text_from_correct_vectors = tfidf_matrix[len(original_text):2*len(original_text)]
            mimicking_text_from_incorrect_vectors = tfidf_matrix[2*len(original_text):]
            
            # Compute metric between original and mimicking from original
            similarity_scores = cosine_similarity(original_vectors, mimicking_text_from_correct_vectors) # compute element-wise on the row of the matrix
            original_mimicking_from_correct = np.mean(similarity_scores) 

            # Compute metric between original and mimicking from obfuscation
            similarity_scores = cosine_similarity(original_vectors, mimicking_text_from_incorrect_vectors)
            original_mimicking_from_incorrect = np.mean(similarity_scores)
            cosine_similarity_logs[speaker] ={"original_mimicking_from_correct": original_mimicking_from_correct, "original_mimicking_from_incorrect": original_mimicking_from_incorrect}
            
        return cosine_similarity_logs


summarize_cosine_similarity_logs = {}  
for dataset_name in  ['speech', 'quora']: 
    summarize_cosine_similarity_logs[dataset_name] ={}         
    for api in  ['deepseek', '4o-mini', 'o3-mini', 'gemini']:
        summarize_cosine_similarity_logs[dataset_name][api] ={}
        for with_without in ['with_user_metadata', 'without_user_metadata']:
            result = compute_vocabulary_features(api=api, dataset_name=dataset_name, with_without = with_without)
            print(result)
            summarize_cosine_similarity_logs[dataset_name][api][with_without] = result

# save to json file
with open('/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/all_cosine_similarity_logs_verification_mimicking.json', 'w') as f:
    json.dump(summarize_cosine_similarity_logs, f, indent=4)
