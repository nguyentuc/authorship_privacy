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
    root_path = '/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/without_user_metadata/'
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

        print("Obfuscation Correct:")
        obf_texts = [word_tokenize(doc.lower()) for doc in obfuscation_correct]
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


        print("Obfuscation Incorrect:")
        mimicking_from_ori = [word_tokenize(doc.lower()) for doc in obfuscation_incorrect]
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
    

LDA()
