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


def sentiment_analysis():
    root_path = '/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/without_user_metadata/'
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    print(f"Dataset structure: {dataset}")

    # Load the sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

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

        # Make predictions
        predictions = sentiment_pipeline(original_text)
        # Display results
        original = []
        for pred in predictions:
            original.append(pred['label'])
        pos = original.count("POSITIVE")
        print(f"Original HS: {pos/len(original)}")
        
        # load obfuscation text
        df = pd.read_csv(root_path+'obfuscation_from_correct_attribute/'+speaker+'.csv')
        obfuscation_correct = []
        for index, row in df.iterrows():
            obfuscation_correct.append(row['Obfuscation'])
        
        # Make predictions
        predictions = sentiment_pipeline(obfuscation_correct)
        # Display results
        obfuscations = []
        for pred in predictions:
            obfuscations.append(pred['label'])
        pos = obfuscations.count("POSITIVE")
        print(f"obfuscations Correct HS: {pos/len(original)}")
        

        # text mimicking from original 
        df = pd.read_csv(root_path+'obfuscation_from_incorrect_attribute/'+speaker+'.csv')
        obfuscation_incorrect = []
        for index, row in df.iterrows():
            obfuscation_incorrect.append(row['Obfuscation'])
        
        # Make predictions
        predictions = sentiment_pipeline(obfuscation_incorrect)
        # Display results
        mimic_ori = []
        for pred in predictions:
            mimic_ori.append(pred['label'])
        pos = mimic_ori.count("POSITIVE")
        print(f"Obfuscatio Incorrect HS: {pos/len(original)}")
    

sentiment_analysis()
