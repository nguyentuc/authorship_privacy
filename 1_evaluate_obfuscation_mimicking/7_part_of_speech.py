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

root_path = '/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/without_user_metadata/'

def calculate_pos():
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
        df = pd.read_csv(root_path+'obfuscation/'+speaker+'.csv')
        obfuscation_text = []
        for index, row in df.iterrows():
            obfuscation_text.append(row['Obfuscation'])
        pos_count_obs = pos_extraction(obfuscation_text)
        print(f"POS Count Obfuscation: {pos_count_obs}")


        # text mimicking from original 
        df = pd.read_csv(root_path+'mimicking_from_original/'+speaker+'.csv')
        mimicking_text_from_ori = []
        for index, row in df.iterrows():
            mimicking_text_from_ori.append(row['Mimicking'])
        pos_count_mimicking_ori = pos_extraction(mimicking_text_from_ori)
        print(f"POS Count mimicking from Original: {pos_count_mimicking_ori}")


        # text mimicking from obfuscation
        df = pd.read_csv(root_path+'mimicking_from_obfuscation/'+speaker+'.csv')
        mimick_obf = []
        for index, row in df.iterrows():
            mimick_obf.append(row['Mimicking'])
        pos_count_mimicking_obf = pos_extraction(mimick_obf)
        print(f"POS Count mimicking from Obfus: {pos_count_mimicking_obf}")

# calculate_pos()

def visualization_pos_stat():

    # POS statistics for each speaker with metadata
    # pos_stats = {
    #     "Obama": {
    #         "Ori": {'NOUN': 2850, 'VERB': 1853, 'PRON': 1711, 'PUNCT': 1592, 'ADP': 1567, 'DET': 1247, 'ADJ': 1030, 'AUX': 990, 'CCONJ': 671, 'ADV': 592, 'PROPN': 585, 'PART': 509, 'SCONJ': 407, 'NUM': 110, 'INTJ': 13, 'SYM': 4, 'X': 1},
    #         "Obfuscation": {'NOUN': 3223, 'VERB': 2269, 'PUNCT': 1717, 'PRON': 1412, 'ADP': 1271, 'DET': 1159, 'ADJ': 938, 'AUX': 705, 'CCONJ': 609, 'ADV': 529, 'PART': 368, 'SCONJ': 334, 'PROPN': 155, 'NUM': 26, 'INTJ': 10},
    #         "Mimic_Ori": {'NOUN': 2954, 'VERB': 2068, 'PUNCT': 1740, 'PRON': 1697, 'ADP': 1361, 'DET': 1178, 'ADJ': 808, 'AUX': 751, 'CCONJ': 589, 'ADV': 581, 'PART': 455, 'SCONJ': 354, 'PROPN': 181, 'NUM': 47, 'INTJ': 9},
    #         "Mimic_Obfus": {'NOUN': 3232, 'VERB': 2235, 'PUNCT': 1756, 'PRON': 1670, 'ADP': 1423, 'DET': 1256, 'ADJ': 885, 'AUX': 676, 'CCONJ': 574, 'ADV': 540, 'PART': 370, 'SCONJ': 339, 'PROPN': 176, 'NUM': 40, 'INTJ': 9},
    #     },
    #     "Bush": {
    #         "Ori": {'NOUN': 1432, 'VERB': 809, 'PUNCT': 739, 'ADP': 709, 'DET': 600, 'PRON': 545, 'ADJ': 539, 'AUX': 397, 'PROPN': 354, 'CCONJ': 311, 'ADV': 218, 'PART': 199, 'SCONJ': 107, 'NUM': 48, 'SYM': 9, 'INTJ': 1, 'X': 1},
    #         "Obfuscation": {'NOUN': 1689, 'VERB': 1049, 'PUNCT': 871, 'ADP': 625, 'ADJ': 550, 'DET': 547, 'PRON': 524, 'CCONJ': 301, 'AUX': 288, 'ADV': 233, 'PART': 152, 'SCONJ': 131, 'PROPN': 105, 'NUM': 15, 'X': 1},
    #         "Mimic_Ori": {'NOUN': 1560, 'VERB': 970, 'PUNCT': 769, 'PRON': 653, 'ADP': 645, 'DET': 540, 'ADJ': 475, 'AUX': 315, 'CCONJ': 304, 'ADV': 228, 'PART': 202, 'SCONJ': 147, 'PROPN': 139, 'NUM': 16, 'X': 1},
    #         "Mimic_Obfus": {'NOUN': 1616, 'VERB': 1066, 'PUNCT': 832, 'PRON': 704, 'ADP': 676, 'DET': 543, 'ADJ': 479, 'AUX': 334, 'CCONJ': 295, 'ADV': 243, 'PART': 161, 'SCONJ': 138, 'PROPN': 110, 'NUM': 17, 'X': 1},
    #     },
    #     "Trump": {
    #         "Ori": {'NOUN': 1157, 'PUNCT': 1000, 'PRON': 970, 'VERB': 944, 'ADP': 691, 'DET': 629, 'AUX': 562, 'ADJ': 480, 'ADV': 392, 'CCONJ': 326, 'PROPN': 320, 'PART': 214, 'SCONJ': 193, 'NUM': 85, 'INTJ': 12, 'SYM': 9},
    #         "Obfuscation": {'NOUN': 1781, 'VERB': 1345, 'PUNCT': 1079, 'PRON': 845, 'ADP': 713, 'ADJ': 627, 'DET': 590, 'AUX': 444, 'CCONJ': 382, 'ADV': 360, 'PART': 212, 'SCONJ': 166, 'PROPN': 107, 'NUM': 42, 'INTJ': 7, 'SYM': 4},
    #         "Mimic_Ori": {'NOUN': 1448, 'PUNCT': 1293, 'VERB': 1263, 'PRON': 1109, 'ADP': 663, 'DET': 615, 'AUX': 589, 'ADJ': 549, 'ADV': 446, 'CCONJ': 339, 'PART': 250, 'SCONJ': 181, 'PROPN': 170, 'NUM': 36, 'INTJ': 9, 'SYM': 5},
    #         "Mimic_Obfus": {'NOUN': 1669, 'VERB': 1346, 'PUNCT': 1163, 'PRON': 1084, 'ADP': 735, 'DET': 687, 'ADJ': 550, 'AUX': 493, 'ADV': 381, 'CCONJ': 359, 'PART': 220, 'SCONJ': 172, 'PROPN': 148, 'NUM': 46, 'INTJ': 11, 'SYM': 5},
    #     }
    # }

    pos_stats = {
        "Obama": {
            "Ori": {'NOUN': 2850, 'VERB': 1853, 'PRON': 1711, 'PUNCT': 1592, 'ADP': 1567, 'DET': 1247, 'ADJ': 1030, 'AUX': 990, 'CCONJ': 671, 'ADV': 592, 'PROPN': 585, 'PART': 509, 'SCONJ': 407, 'NUM': 110, 'INTJ': 13, 'SYM': 4, 'X': 1},
            "Obfuscation": {'NOUN': 3262, 'VERB': 2308, 'PUNCT': 1723, 'PRON': 1336, 'ADP': 1294, 'DET': 1077, 'ADJ': 1016, 'AUX': 674, 'CCONJ': 562, 'ADV': 545, 'PART': 348, 'SCONJ': 326, 'PROPN': 151, 'NUM': 31, 'INTJ': 9},
            "Mimic_Ori": {'NOUN': 3042, 'VERB': 2079, 'PUNCT': 1764, 'PRON': 1581, 'ADP': 1291, 'DET': 1210, 'ADJ': 807, 'AUX': 749, 'ADV': 603, 'CCONJ': 594, 'PART': 433, 'SCONJ': 344, 'PROPN': 176, 'NUM': 46, 'INTJ': 9},
            "Mimic_Obfus": {'NOUN': 3388, 'VERB': 2292, 'PUNCT': 1830, 'PRON': 1426, 'ADP': 1388, 'DET': 1179, 'ADJ': 946, 'AUX': 629, 'CCONJ': 564, 'ADV': 549, 'PART': 330, 'SCONJ': 307, 'PROPN': 154, 'NUM': 38, 'INTJ': 9},
        },
        "Bush": {
            "Ori": {'NOUN': 1432, 'VERB': 809, 'PUNCT': 739, 'ADP': 709, 'DET': 600, 'PRON': 545, 'ADJ': 539, 'AUX': 397, 'PROPN': 354, 'CCONJ': 311, 'ADV': 218, 'PART': 199, 'SCONJ': 107, 'NUM': 48, 'SYM': 9, 'INTJ': 1, 'X': 1},
            "Obfuscation": {'NOUN': 1696, 'VERB': 1073, 'PUNCT': 871, 'ADP': 635, 'ADJ': 565, 'DET': 510, 'PRON': 504, 'CCONJ': 305, 'AUX': 288, 'ADV': 231, 'PART': 171, 'SCONJ': 112, 'PROPN': 107, 'NUM': 14, 'X': 1},
            "Mimic_Ori": {'NOUN': 1624, 'VERB': 951, 'PUNCT': 778, 'ADP': 658, 'DET': 598, 'PRON': 570, 'ADJ': 518, 'AUX': 303, 'CCONJ': 292, 'ADV': 222, 'PART': 191, 'SCONJ': 141, 'PROPN': 113, 'NUM': 15, 'X': 1},
            "Mimic_Obfus": {'NOUN': 1662, 'VERB': 1057, 'PUNCT': 836, 'ADP': 643, 'PRON': 604, 'ADJ': 579, 'DET': 533, 'CCONJ': 320, 'AUX': 291, 'ADV': 258, 'PART': 185, 'SCONJ': 125, 'PROPN': 114, 'NUM': 14, 'X': 1, 'SYM': 1},
        },
        "Trump": {
            "Ori": {'NOUN': 1157, 'PUNCT': 1000, 'PRON': 970, 'VERB': 944, 'ADP': 691, 'DET': 629, 'AUX': 562, 'ADJ': 480, 'ADV': 392, 'CCONJ': 326, 'PROPN': 320, 'PART': 214, 'SCONJ': 193, 'NUM': 85, 'INTJ': 12, 'SYM': 9},
            "Obfuscation": {'NOUN': 1822, 'VERB': 1320, 'PUNCT': 1086, 'PRON': 760, 'ADP': 734, 'ADJ': 658, 'DET': 657, 'AUX': 415, 'ADV': 374, 'CCONJ': 341, 'PART': 194, 'SCONJ': 170, 'PROPN': 101, 'NUM': 40, 'INTJ': 7, 'SYM': 5},
            "Mimic_Ori": {'NOUN': 1702, 'VERB': 1184, 'PUNCT': 1073, 'PRON': 908, 'DET': 793, 'ADP': 786, 'ADJ': 556, 'AUX': 448, 'ADV': 394, 'CCONJ': 308, 'PART': 205, 'SCONJ': 184, 'PROPN': 128, 'NUM': 48, 'INTJ': 7, 'SYM': 5},
            "Mimic_Obfus": {'NOUN': 1851, 'VERB': 1371, 'PUNCT': 1145, 'PRON': 943, 'ADP': 764, 'DET': 737, 'ADJ': 507, 'AUX': 389, 'ADV': 389, 'CCONJ': 361, 'PART': 206, 'SCONJ': 204, 'PROPN': 108, 'NUM': 48, 'INTJ': 8, 'SYM': 5},
        }
    }

    # Plot POS distributions for each speaker
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    colors = ['blue', 'orange', 'green', 'red']
    datasets = ['Ori', 'Obfuscation', 'Mimic_Ori', 'Mimic_Obfus']

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
    plt.savefig(root_path+'pos_distribtution.png')
    plt.show()

visualization_pos_stat()