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
        root_path = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/additional_experiment/{dataset_name}/{api}/{with_without}/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        print(f"Dataset structure: {dataset}")

        # load all the text need to compute

        speakers = ['obama', 'bush', 'trump']
        word_diversity = {}
        for speaker in speakers:
            word_diversity[speaker] = {}

            # original text
            # sample text that has bigger than 50 words
            author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            original_text = [example['text'] for example in author_dataset]
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
            word_diversity[speaker]['original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}

            # load obfuscation text
            df = pd.read_csv(root_path+speaker+'.csv')
            obfuscation_text = []
            for index, row in df.iterrows():
                obfuscation_text.append(row['Obfuscation'])
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text)
            word_diversity[speaker]['obfuscation'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}

            # text mimicking from original 
            # df = pd.read_csv(root_path+'mimicking_from_original/'+speaker+'.csv')
            # mimicking_text_from_ori = []
            # for index, row in df.iterrows():
            #     mimicking_text_from_ori.append(row['Mimicking'])
            # vocab_size, avg_length, diversity = vocabulary_size_and_diversity(mimicking_text_from_ori)
            # word_diversity[speaker]['moriginal'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}

            # text mimicking from obfuscation
            # df = pd.read_csv(root_path+'mimicking_from_obfuscation/'+speaker+'.csv')
            # mimick_obf = []
            # for index, row in df.iterrows():
            #     mimick_obf.append(row['Mimicking'])
            # vocab_size, avg_length, diversity = vocabulary_size_and_diversity(mimick_obf)
            # word_diversity[speaker]['mobfuscation'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}
        return word_diversity


    else: # processing for quora
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        word_diversity = {}
        for filename in os.listdir(root_path):
            speaker = filename.split('.')[0]
            word_diversity[speaker] = {}

            # load and compute ppl on original text
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)
            original_text = [example['Question']+' '+example['Answer'] for idx, example in author_dataset.iterrows() ]

            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(original_text)
            word_diversity[speaker]['original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}


            # load obfuscation text (from 20 percent of original text)
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/additional_experiment/{dataset_name}/{api}/{with_without}/'+speaker+'.csv')
            obfuscation_text = []
            for index, row in df.iterrows():
                obfuscation_text.append(row['Obfuscation'].replace('\n', " "))
            vocab_size, avg_length, diversity = vocabulary_size_and_diversity(obfuscation_text)
            word_diversity[speaker]['obfuscation'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}


            # text mimicking from original 
            # df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/{with_without}/mimicking_from_original/'+speaker+'.csv')
            # mimicking_text_from_ori = []
            # for index, row in df.iterrows():
            #     mimicking_text_from_ori.append(row['Mimicking'])
            # vocab_size, avg_length, diversity = vocabulary_size_and_diversity(mimicking_text_from_ori)
            # word_diversity[speaker]['m_original'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}


            # text mimicking from obfuscation
            # df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/{with_without}/mimicking_from_obfuscation/'+speaker+'.csv')
            # mimick_obf = []
            # for index, row in df.iterrows():
            #     mimick_obf.append(row['Mimicking'].replace('\n', " "))  
            # vocab_size, avg_length, diversity = vocabulary_size_and_diversity(mimick_obf)
            # word_diversity[speaker]['m_obfuscation'] ={"vocab_size": vocab_size, "avg_length": avg_length, "diversity": diversity}
        return word_diversity


summarize_word_diversity = {}  
for dataset_name in  ['speech', 'quora']: 
    summarize_word_diversity[dataset_name] ={}         
    for api in  ['deepseek', '4o-mini', 'o3-mini', 'gemini']:
        summarize_word_diversity[dataset_name][api] ={}
        for with_without in ['with', 'without']:
            result = compute_vocabulary_features(api=api, dataset_name=dataset_name, with_without = with_without)
            print(result)
            summarize_word_diversity[dataset_name][api][with_without] = result

# save to json file
with open('/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/additional_experiment/all_word_diversity_logs.json', 'w') as f:
    json.dump(summarize_word_diversity, f, indent=4)