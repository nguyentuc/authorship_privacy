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

# Return vocab size, average length of text, text diversity
def pos_extraction(documents):
    nlp = spacy.load('en_core_web_sm')
    # Initialize Counter
    pos_counts = Counter()
    # Process each document and compute POS tags
    for doc in documents:
        spacy_doc = nlp(doc)
        pos_counts.update(token.pos_ for token in spacy_doc)
    return pos_counts


def calculate_pos():
    root_path = '/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/without_user_metadata/'
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
        pos_count_ori = pos_extraction(original_text)
        print(f"POS Count Ori: {pos_count_ori}")

        # load obfuscation text
        # obfuscation from correct attribute
        df = pd.read_csv(root_path+'obfuscation_from_correct_attribute/'+speaker+'.csv')
        obfuscation_text = []
        for index, row in df.iterrows():
            obfuscation_text.append(row['Obfuscation'])

        pos_count_obs = pos_extraction(obfuscation_text)
        print(f"POS Correct: {pos_count_obs}")

        # obfuscation from incorrect 
        df = pd.read_csv(root_path+'obfuscation_from_incorrect_attribute/'+speaker+'.csv')
        mimicking_text_from_ori = []
        for index, row in df.iterrows():
            mimicking_text_from_ori.append(row['Obfuscation'])

        pos_count_mimicking_ori = pos_extraction(mimicking_text_from_ori)
        print(f"POS Incorrect: {pos_count_mimicking_ori}")

calculate_pos()


def visualization_pos_stat():

    # POS statistics for each speaker
    pos_stats = {
        "Obama": {
            "Ori": {'NOUN': 2850, 'VERB': 1853, 'PRON': 1711, 'PUNCT': 1592, 'ADP': 1567, 'DET': 1247, 'ADJ': 1030, 'AUX': 990, 'CCONJ': 671, 'ADV': 592, 'PROPN': 585, 'PART': 509, 'SCONJ': 407, 'NUM': 110, 'INTJ': 13, 'SYM': 4, 'X': 1},
            "Correct": {'NOUN': 3131, 'VERB': 2316, 'PUNCT': 1793, 'PRON': 1467, 'ADP': 1256, 'DET': 1100, 'ADJ': 948, 'AUX': 642, 'ADV': 584, 'CCONJ': 573, 'PART': 359, 'SCONJ': 322, 'PROPN': 150, 'NUM': 38, 'INTJ': 9},
            "InCorrect": {'NOUN': 3221, 'VERB': 2270, 'PUNCT': 1818, 'PRON': 1447, 'ADP': 1233, 'DET': 1034, 'ADJ': 923, 'AUX': 697, 'CCONJ': 580, 'ADV': 575, 'PART': 377, 'SCONJ': 323, 'PROPN': 161, 'NUM': 34, 'INTJ': 9},
             },
        "Bush": {
            "Ori": {'NOUN': 1432, 'VERB': 809, 'PUNCT': 739, 'ADP': 709, 'DET': 600, 'PRON': 545, 'ADJ': 539, 'AUX': 397, 'PROPN': 354, 'CCONJ': 311, 'ADV': 218, 'PART': 199, 'SCONJ': 107, 'NUM': 48, 'SYM': 9, 'INTJ': 1, 'X': 1},
            "Correct": {'NOUN': 1652, 'VERB': 1050, 'PUNCT': 844, 'ADP': 632, 'ADJ': 566, 'PRON': 536, 'DET': 514, 'AUX': 302, 'CCONJ': 286, 'ADV': 234, 'PART': 157, 'SCONJ': 136, 'PROPN': 107, 'NUM': 13, 'X': 1},
            "InCorrect": {'NOUN': 1672, 'VERB': 1066, 'PUNCT': 846, 'ADP': 658, 'ADJ': 558, 'DET': 533, 'PRON': 517, 'AUX': 283, 'CCONJ': 272, 'ADV': 221, 'PART': 180, 'SCONJ': 130, 'PROPN': 102, 'NUM': 18, 'X': 1, 'SYM': 1},
        },
        "Trump": {
            "Ori": {'NOUN': 1157, 'PUNCT': 1000, 'PRON': 970, 'VERB': 944, 'ADP': 691, 'DET': 629, 'AUX': 562, 'ADJ': 480, 'ADV': 392, 'CCONJ': 326, 'PROPN': 320, 'PART': 214, 'SCONJ': 193, 'NUM': 85, 'INTJ': 12, 'SYM': 9},
            "Correct": {'NOUN': 1795, 'VERB': 1303, 'PUNCT': 1087, 'PRON': 830, 'ADP': 745, 'DET': 642, 'ADJ': 597, 'AUX': 447, 'CCONJ': 372, 'ADV': 361, 'PART': 215, 'SCONJ': 137, 'PROPN': 115, 'NUM': 41, 'INTJ': 8, 'SYM': 5},
            "InCorrect": {'NOUN': 1753, 'VERB': 1321, 'PUNCT': 1095, 'PRON': 854, 'ADP': 725, 'ADJ': 664, 'DET': 605, 'AUX': 441, 'ADV': 374, 'CCONJ': 368, 'PART': 201, 'SCONJ': 171, 'PROPN': 105, 'NUM': 36, 'INTJ': 7, 'SYM': 4},
        }
    }

    # Plot POS distributions for each speaker
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    colors = ['blue', 'orange', 'green']
    datasets = ['Ori', 'Correct', 'InCorrect']

    for idx, (speaker, data) in enumerate(pos_stats.items()):
        pos_tags = list(data['Ori'].keys())  # Get POS categories
        x = np.arange(len(pos_tags))  # X-axis positions

        # Plot each dataset
        for i, dataset in enumerate(datasets):
            values = [data[dataset].get(pos, 0) for pos in pos_tags]
            axes[idx].bar(x + i * 0.2, values, width=0.2, label=dataset, color=colors[i])

        axes[idx].set_title(f"POS Distribution - {speaker}")
        axes[idx].set_xticks(x + 0.3)
        axes[idx].set_xticklabels(pos_tags, rotation=45, ha='right')
        axes[idx].legend()

    # Set shared labels
    fig.suptitle("POS Tag Distribution Across Different Speech Datasets", fontsize=14)
    fig.supylabel("Frequency Count")
    plt.tight_layout()
    plt.savefig('pos_distribtution.png')
    plt.show()

visualization_pos_stat()