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
    api='deepseek'
    dataset='speech'
    root_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/'
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

        # Make predictions
        predictions = sentiment_pipeline(original_text)
        # Display results
        original = []
        for pred in predictions:
            original.append(pred['label'])
        pos = original.count("POSITIVE")
        print(f"Original HS: {pos/len(original)}")
        
        # load obfuscation text
        df = pd.read_csv(root_path+'obfuscation/'+speaker+'.csv')
        obfuscation_text = []
        for index, row in df.iterrows():
            obfuscation_text.append(row['Obfuscation'])
        
        # Make predictions
        predictions = sentiment_pipeline(obfuscation_text)
        # Display results
        obfuscations = []
        for pred in predictions:
            obfuscations.append(pred['label'])
        pos = obfuscations.count("POSITIVE")
        print(f"obfuscations HS: {pos/len(original)}")
        

        # text mimicking from original 
        df = pd.read_csv(root_path+'mimicking_from_original/'+speaker+'.csv')
        mimicking_text_from_ori = []
        for index, row in df.iterrows():
            mimicking_text_from_ori.append(row['Mimicking'])
        
        # Make predictions
        predictions = sentiment_pipeline(mimicking_text_from_ori)
        # Display results
        mimic_ori = []
        for pred in predictions:
            mimic_ori.append(pred['label'])
        pos = mimic_ori.count("POSITIVE")
        print(f"mimic_ori HS: {pos/len(original)}")

        # text mimicking from obfuscation
        df = pd.read_csv(root_path+'mimicking_from_obfuscation/'+speaker+'.csv')
        mimick_obf = []
        for index, row in df.iterrows():
            mimick_obf.append(row['Mimicking'])
        predictions = sentiment_pipeline(mimick_obf)
        # Display results
        mimic_obfus = []
        for pred in predictions:
            mimic_obfus.append(pred['label'])
        
        pos = mimic_obfus.count("POSITIVE")
        print(f"mimic_obfus HS: {pos/len(original)}")
    

sentiment_analysis()
