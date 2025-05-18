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
        root_path = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        print(f"Dataset structure: {dataset}")

        # load all the text need to compute

        speakers = ['obama', 'bush', 'trump']
        cosine_similarity_logs = {}
        for speaker in speakers:
            cosine_similarity_logs[speaker] = {}

            # original text
            # sample text that has bigger than 50 words
            author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            original_text = [example['text'] for example in author_dataset]
            
            # text obfuscation from original 
            df = pd.read_csv(root_path+'obfuscation_from_original/'+speaker+'.csv')
            obfuscation_text_from_ori = []
            for index, row in df.iterrows():
                obfuscation_text_from_ori.append(row['Obfuscation'].replace('\n', ' '))
            
            # text obfuscation from mimic
            df = pd.read_csv(root_path+'obfuscation_from_mimic/'+speaker+'.csv')
            obfuscation_from_mimic = []
            for index, row in df.iterrows():
                obfuscation_from_mimic.append(row['Obfuscation'].replace("\n", ''))

            
            # Combine both corpora for consistent TF-IDF vectorization
            all_texts = original_text + obfuscation_text_from_ori + obfuscation_from_mimic

            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Split back into original and paraphrased matrices
            original_vectors = tfidf_matrix[:len(original_text)]
            obfuscation_text_from_ori_vectors = tfidf_matrix[len(original_text):2*len(original_text)]
            obfuscation_from_mimic_vectors = tfidf_matrix[2*len(original_text):]
            
            # Compute metric between original and obfuscation from original
            similarity_scores = cosine_similarity(original_vectors, obfuscation_text_from_ori_vectors) # compute element-wise on the row of the matrix
            original_obuscation_from_original = np.mean(similarity_scores) 

            # Compute metric between original and obfuscation from mimicking
            similarity_scores = cosine_similarity(original_vectors, obfuscation_from_mimic_vectors)
            original_obfuscation_from_mimicking = np.mean(similarity_scores)
            cosine_similarity_logs[speaker] ={"original_obfuscation_from_original": original_obuscation_from_original, "original_obfuscation_from_mimicking": original_obfuscation_from_mimicking}
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
            author_dataset = author_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)
            original_text = [example['Question']+' '+example['Answer'] for idx, example in author_dataset.iterrows() ]
        
            # text obfuscation from original 
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/obfuscation_from_original/'+speaker+'.csv')
            obfuscation_text_from_ori = []
            for index, row in df.iterrows():
                obfuscation_text_from_ori.append(row['Obfuscation'].replace("\n", " "))
            
            # text obfuscation from mimicking
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/obfuscation_from_mimic/'+speaker+'.csv')
            obf_from_mimicking = []
            for index, row in df.iterrows():
                obf_from_mimicking.append(row['Obfuscation'].replace('\n', " "))  
            
            # Combine both corpora for consistent TF-IDF vectorization
            all_texts = original_text + obfuscation_text_from_ori + obf_from_mimicking

            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Split back into original and paraphrased matrices
            original_vectors = tfidf_matrix[:len(original_text)]
            obfuscation_text_from_ori_vectors = tfidf_matrix[len(original_text):2*len(original_text)]
            obf_from_mimicking_vectors = tfidf_matrix[2*len(original_text):]
            
            # Compute metric between original and mimicking from original
            similarity_scores = cosine_similarity(original_vectors, obfuscation_text_from_ori_vectors) # compute element-wise on the row of the matrix
            original_obfuscation_from_original = np.mean(similarity_scores) 

            # Compute metric between original and mimicking from obfuscation
            similarity_scores = cosine_similarity(original_vectors, obf_from_mimicking_vectors)
            original_obfuscation_from_mimicking = np.mean(similarity_scores)
            cosine_similarity_logs[speaker] ={"original_obfuscation_from_original": original_obfuscation_from_original, "original_obfuscation_from_mimicking": original_obfuscation_from_mimicking}
            
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
with open('/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/all_cosine_similarity_logs_mimicking_obfuscation.json', 'w') as f:
    json.dump(summarize_cosine_similarity_logs, f, indent=4)
