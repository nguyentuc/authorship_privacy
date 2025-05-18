import os
import json
import pandas as pd
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Return vocab size, average length of text, text diversity
def vocabulary_size_and_diversity(texts):
    tokens = []
    length = []
    for text in texts:
        length.append(len(word_tokenize(text.lower())))
        tokens.extend(word_tokenize(text.lower()))

    # get how many words are use
    total = len(tokens)
    # Get unique tokens (vocabulary)
    vocab = set(tokens)
    vocab_size = len(vocab)
    return vocab_size, np.mean(length), vocab_size/total

data_name = 'speech'
api = 'deepseek'
root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/with_user_metadata/'
dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
print(f"Dataset structure: {dataset}")

# load all the text need to compute

speakers = ['obama', 'bush', 'trump']
mean_diversity =[]
for speaker in speakers:
    print(f"{speaker}")
    # original text
    author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
    author_dataset = author_dataset.shuffle(seed=2024)
    author_dataset = author_dataset.shuffle(seed=2025)
    author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
    original_text = [example['text'] for example in author_dataset]

    vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
    # print(f"Original: {vocab_size}: {avg_length} : {diversity}")

    # obfuscation from correct attribute
    df = pd.read_csv(root_path+'obfuscation_from_correct_attribute/'+speaker+'.csv')
    obfuscation_text = []
    for index, row in df.iterrows():
        obfuscation_text.append(row['Obfuscation'])

    vocab_size, avg_length, diversity1 = vocabulary_size_and_diversity(obfuscation_text)
    # print(f"Obfuscation from Correct: {vocab_size}: {avg_length} : {diversity}")


    # obfuscation from incorrect 
    df = pd.read_csv(root_path+'obfuscation_from_incorrect_attribute/'+speaker+'.csv')
    mimicking_text_from_ori = []
    for index, row in df.iterrows():
        mimicking_text_from_ori.append(row['Obfuscation'])

    vocab_size, avg_length, diversity2 = vocabulary_size_and_diversity(mimicking_text_from_ori)
    # print(f"Obfuscation from Incorrect: {vocab_size}: {avg_length} : {diversity}")
    avg_diversity = (diversity1 + diversity2) /2
    mean_diversity.append(avg_diversity)

print("Avg diversity:", np.mean(mean_diversity))


    