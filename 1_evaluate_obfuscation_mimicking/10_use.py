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
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub
import numpy as np
import torch.nn.functional as F
import tensorflow as tf

# Load the Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def USE():
    dataset='speech'
    api = 'gemini'
    root_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/without_user_metadata/'
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    print(f"Dataset structure: {dataset}")

    # Load the sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

    # load all the text need to compute
    speakers = ['obama', 'bush', 'trump']
    for speaker in speakers:
        print(f"{speaker}")
        # original text
        # sample text that has bigger than 50 words
        author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]

        # load obfuscation text
        df = pd.read_csv(root_path+'obfuscation/'+speaker+'.csv')
        obfuscation_text = []
        for index, row in df.iterrows():
            obfuscation_text.append(row['Obfuscation'])
        
        # text mimicking from original 
        df = pd.read_csv(root_path+'mimicking_from_original/'+speaker+'.csv')
        mimicking_text_from_ori = []
        for index, row in df.iterrows():
            mimicking_text_from_ori.append(row['Mimicking'])
        
        # text mimicking from obfuscation
        df = pd.read_csv(root_path+'mimicking_from_obfuscation/'+speaker+'.csv')
        mimick_obf = []
        for index, row in df.iterrows():
            mimick_obf.append(row['Mimicking'])
    
        # compute embedding with USE for each text document
        original_embeddings = model(original_text)
        obfuscation_embeddings = model(obfuscation_text)
        mimick_ori_embeddings = model(mimicking_text_from_ori)
        mimick_obf_embeddings = model(mimick_obf)

        # compute cosine similarity
        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_obfuscation_embeddings = tf.nn.l2_normalize(obfuscation_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(norm_original_embeddings, norm_obfuscation_embeddings, transpose_b=True)  # (5, 512) . (5, 512).T -> (5, 5)
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Sim Ori-Obfus: {mean_similarity}")

        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_mimick_ori_embeddings = tf.nn.l2_normalize(mimick_ori_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(norm_original_embeddings, norm_mimick_ori_embeddings, transpose_b=True)  # (5, 512) . (5, 512).T -> (5, 5)
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Sim Ori-Mimick from Ori: {mean_similarity}")


        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_mimick_obf_embeddings = tf.nn.l2_normalize(mimick_obf_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(norm_original_embeddings, norm_mimick_obf_embeddings, transpose_b=True)  # (5, 512) . (5, 512).T -> (5, 5)
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Sim Ori-Mimick from Obfus: {mean_similarity}")

USE()
