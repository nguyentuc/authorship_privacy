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


def compute_vocabulary_features(api, dataset_name, with_without):

    if dataset_name == "speech":
        root_path = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        print(f"Dataset structure: {dataset}")

        # load all the text need to compute

        speakers = ['obama', 'bush', 'trump']
        word_diversity = {}
        for speaker in speakers:
            word_diversity[speaker] = {}

            # original text
            author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            original_text = [example['text'] for example in author_dataset]
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
            word_diversity[speaker]['original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}

            # text obfuscation from original 
            df = pd.read_csv(root_path+'obfuscation_from_original/'+speaker+'.csv')
            obfuscation_text_from_ori = []
            for index, row in df.iterrows():
                obfuscation_text_from_ori.append(row['Obfuscation'].replace('\n', ' '))
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text_from_ori)
            word_diversity[speaker]['obfuscation_from_original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}

            # text obfuscation from mimic
            df = pd.read_csv(root_path+'obfuscation_from_mimic/'+speaker+'.csv')
            obfuscation_from_mimic = []
            for index, row in df.iterrows():
                obfuscation_from_mimic.append(row['Obfuscation'].replace("\n", ''))
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_from_mimic)
            word_diversity[speaker]['obfuscation_from_mimicking'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}
        return word_diversity


    else: # processing for quora
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        word_diversity = {}
        for filename in os.listdir(root_path):
            speaker = filename.split('.')[0]
            word_diversity[speaker] = {}

            # original text
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)
            original_text = [example['Question']+' '+example['Answer'] for idx, example in author_dataset.iterrows() ]
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
            word_diversity[speaker]['original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}


            # text obfuscation from original 
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/obfuscation_from_original/'+speaker+'.csv')
            obfuscation_text_from_ori = []
            for index, row in df.iterrows():
                obfuscation_text_from_ori.append(row['Obfuscation'].replace("\n", " "))
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text_from_ori)
            word_diversity[speaker]['obfuscation_from_original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}


            # text obfuscation from mimicking
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/{with_without}/obfuscation_from_mimic/'+speaker+'.csv')
            obf_from_mimicking = []
            for index, row in df.iterrows():
                obf_from_mimicking.append(row['Obfuscation'].replace('\n', " "))  
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obf_from_mimicking)
            word_diversity[speaker]['obfuscation_from_mimicking'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}
        return word_diversity


summarize_word_diversity = {}  
for dataset_name in  ['speech', 'quora']: 
    summarize_word_diversity[dataset_name] ={}         
    for api in  ['deepseek', '4o-mini', 'o3-mini', 'gemini']:
        summarize_word_diversity[dataset_name][api] ={}
        for with_without in ['with_user_metadata', 'without_user_metadata']:
            result = compute_vocabulary_features(api=api, dataset_name=dataset_name, with_without = with_without)
            print(result)
            summarize_word_diversity[dataset_name][api][with_without] = result

# save to json file
with open('/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/all_word_diversity_logs_mimicking_obfuscation.json', 'w') as f:
    json.dump(summarize_word_diversity, f, indent=4)