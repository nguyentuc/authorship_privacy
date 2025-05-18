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
    data_name = 'speech'
    api = 'gemini'
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/with_user_metadata/'
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    print(f"Dataset structure: {dataset}")

    # load all the text need to compute
    speakers = ['obama', 'bush', 'trump']
    for speaker in speakers:
        print(f"Working on {speaker}")
        # original text
        # sample text that has bigger than 50 words
        author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]

        # obfuscate from correct
        df = pd.read_csv(root_path+'obfuscation_from_correct_attribute/'+speaker+'.csv')
        obfuscation_correct = []
        for index, row in df.iterrows():
            obfuscation_correct.append(row['Obfuscation'])

        # text obf from incorrect
        df = pd.read_csv(root_path+'obfuscation_from_incorrect_attribute/'+speaker+'.csv')
        obfuscation_incorrect = []
        for index, row in df.iterrows():
            obfuscation_incorrect.append(row['Obfuscation'])
    
        # compute embedding with USE for each text document
        original_embeddings = model(original_text)
        obfuscation_correct_embeddings = model(obfuscation_correct)
        obfuscation_incorrect_embeddings = model(obfuscation_incorrect)
    

        # compute cosine similarity
        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_obfuscation_embeddings = tf.nn.l2_normalize(obfuscation_correct_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(norm_original_embeddings, norm_obfuscation_embeddings, transpose_b=True)  # (5, 512) . (5, 512).T -> (5, 5)
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Sim Ori-Obfus Correct: {mean_similarity}")

        norm_original_embeddings = tf.nn.l2_normalize(original_embeddings, axis=1)
        norm_mimick_ori_embeddings = tf.nn.l2_normalize(obfuscation_incorrect_embeddings, axis=1)
        cosine_similarity_matrix = tf.matmul(norm_original_embeddings, norm_mimick_ori_embeddings, transpose_b=True)  # (5, 512) . (5, 512).T -> (5, 5)
        diagonal_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        mean_similarity = tf.reduce_mean(diagonal_similarities)
        print(f"Sim Ori-Obfus InCorrect: {mean_similarity}")

USE()
