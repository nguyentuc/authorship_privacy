import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


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


def LDA():
    root_path = '/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/without_user_metadata/'
    dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")

    stop_words = list(set(stopwords.words('english')))
    stop_words.append('and')
    stop_words.append('also')
    stop_words.append('the')
    stop_words.append('to')
    stop_words.append('we')
    stop_words.append('.')
    stop_words.append(',')
    stop_words.append('\'s')
    stop_words.append('--')
    stop_words.append('n\'t')
    stop_words.append('\'ve')
    stop_words.append('\'\'')
    stop_words.append('us')
    stop_words.append(';')
    stop_words.append('\'m')
    stop_words.append('must')
    stop_words.append('like')
    stop_words.append('every')
    stop_words.append("'")
    stop_words.append('\'re')

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
    
        # Original
        print("Original:")
        ori_texts = [word_tokenize(doc.lower()) for doc in original_text]
        texts =[]
        for doc_text in ori_texts:
            doc_t =[]
            for word in doc_text:
                if word not in stop_words:
                    doc_t.append(word)
            texts.append(doc_t)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        num_topics = 2  # Number of topics to extract
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            print(topic)

        print("Obfuscation:")
        obf_texts = [word_tokenize(doc.lower()) for doc in obfuscation_text]
        texts =[]
        for doc_text in obf_texts:
            doc_t =[]
            for word in doc_text:
                if word not in stop_words:
                    doc_t.append(word)
            texts.append(doc_t)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        num_topics = 2  # Number of topics to extract
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            print(topic)


        print("Mimicking from Original:")
        mimicking_from_ori = [word_tokenize(doc.lower()) for doc in mimicking_text_from_ori]
        texts =[]
        for doc_text in mimicking_from_ori:
            doc_t =[]
            for word in doc_text:
                if word not in stop_words:
                    doc_t.append(word)
            texts.append(doc_t)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        num_topics = 2  # Number of topics to extract
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            print(topic)


        print("Mimicking from Obfuscation:")
        mimicking_from_obf = [word_tokenize(doc.lower()) for doc in mimick_obf]
        texts =[]
        for doc_text in mimicking_from_obf:
            doc_t =[]
            for word in doc_text:
                if word not in stop_words:
                    doc_t.append(word)
            texts.append(doc_t)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        num_topics = 2  # Number of topics to extract
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)
        topics = lda_model.print_topics(num_words=10)
        for topic in topics:
            print(topic)

    

LDA()
