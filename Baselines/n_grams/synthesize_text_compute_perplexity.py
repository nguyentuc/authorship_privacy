# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import random 
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.stats import entropy
import json
from scipy.spatial.distance import cdist

def load_dataset_k_user(data_path, k):
    log_prob, train_labels = [], []
    # train
    writing_files_train = os.listdir(data_path+'train_40_5_5/')
    random.seed(2024)
    writing_files_train = random.choices(writing_files_train, k=k)

    # load 10 more synthesize with writing sample
    for writing in writing_files_train:
        synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.8/'+ writing)[:10]
        for idx ,row  in synthesize_writing_only.iterrows():
            # get log prob for each document
            prob_each_document = [float(_.split('+;+')[1]) for _ in row['Log_prob_writing_sample'].split('|;|')]
            log_prob.append(prob_each_document)
            train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with profile
    # for writing in writing_files_train:
    #     synthesize_profile_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_profile_only.iterrows():
    #         # get log prob for each document
    #         prob_each_document = [float(_.split('+;+')[1]) for _ in row['Log_prob_user_profile'].split('|;|')]
    #         log_prob.append(prob_each_document)
    #         train_labels.append(writing.split('.')[0])

    # load 10 more with all
    # for writing in writing_files_train:
    #     synthesize_full = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_full.iterrows():
    #         # get log prob for each document
    #         prob_each_document = [float(_.split('+;+')[1]) for _ in row['Log_prob_full_information'].split('|;|')]
    #         log_prob.append(prob_each_document)
    #         train_labels.append(writing.split('.')[0])
    return log_prob, train_labels

for k in [10]:
    print(f"Visualization for {k} users!!!")
    log_prob, train_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)
    # convert label
    labels_dictionary = {}
    idx = 0
    for label in train_labels:
        if label not in labels_dictionary:
            labels_dictionary[label] = idx
            idx+=1

    # encode new labels
    encoded_train_labels, encoded_eval_labels, encoded_test_labels = [],[], []
    for label in train_labels:
        encoded_train_labels.append(labels_dictionary[label])

    for user_i in range(k):
        additional_data_user_i = log_prob[10*user_i: 10*(user_i+1)]
        additional_label_i = encoded_train_labels[10*user_i: 10*(user_i+1)]
        # compute the perplexity for 10 documents of user i
        pp_user_i =[]
        for _ in range(len(additional_data_user_i)):
            log_prob_sum = np.sum(additional_data_user_i[_])

            N = len(additional_data_user_i[_])  # Number of tokens
            avg_log_prob = log_prob_sum / N

            perplexity = np.exp(-avg_log_prob)
            pp_user_i.append(perplexity)
        print(f"PP-synthesize writing-user {user_i}:", pp_user_i)
        

    
