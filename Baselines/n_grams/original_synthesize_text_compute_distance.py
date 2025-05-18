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
    train_texts, train_labels = [], []
    # train
    writing_files_train = os.listdir(data_path+'train_40_5_5/')
    random.seed(2024)
    writing_files_train = random.choices(writing_files_train, k=k)

    # get json file of user profile
    # for user_profile in writing_files_train:
    #     path = '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/converted_user_profile/'+ user_profile.split('.')[0]+'.json'
    #     # Open the file and load the data
    #     with open(path, 'r') as file:
    #         data = json.load(file)
    #     print(data)

    # original training samples
    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train_40_5_5/'+ writing)
        writing_train = writing_train.sample(n=5, random_state=2024, replace=False)
        for idx ,row  in writing_train.iterrows():
            train_texts.append(row['Answer'])
            train_labels.append(writing.split('.')[0])

    # load 10 more original with writing sample
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/10_more_original_writing/'+ writing)
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with writing sample
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_writing_sample'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with profile
    for writing in writing_files_train:
        synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
        for idx ,row  in synthesize_writing_only.iterrows():
            train_texts.append(row['Answer_with_user_profile'])
            train_labels.append(writing.split('.')[0])

    # load 10 more with all
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_full_information'])
    #         train_labels.append(writing.split('.')[0])
    return train_texts, train_labels

for k in [10]:
    print(f"Visualization for {k} users!!!")
    train_data, train_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)

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

    # Create N-grams using CountVectorizer (unigrams and bigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 4), analyzer='char')  # N-grams with unigrams, bigrams and trigrams

    # Fit and transform the training data
    X_train_ngrams = vectorizer.fit_transform(train_data)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_train_ngrams = max_abs_scaler.fit_transform(X_train_ngrams)

    # compting statistic
    # original data points on the first 5*k record
    original_data = X_train_ngrams[:5*k]
    original_label = encoded_train_labels[:5*k]
    print("Original label:", original_label)

    # additional data points on the first 5*k record
    additional_data = X_train_ngrams[5*k:]
    additional_label = encoded_train_labels[5*k:]
    print("Additional label:", additional_label)

    # for each additional data, compute distance with every original data and get average
    final_list = []
    for user_i in range(k):
        # original data
        original_data_user_i = original_data[5*user_i: 5*(user_i+1)]
        original_dense = original_data_user_i.toarray()

        # additional data from i
        additional_data_user_i = additional_data[10*user_i: 10*(user_i+1)]
        additional_dense = additional_data_user_i.toarray()

        # for each additional datapoint, compute distance to the original data and take average
        distances = cdist(additional_dense, original_dense, metric='euclidean')
        print(f"Avg distance additional datapoint to original datasets {user_i}:", distances.mean(axis=1))
        final_list.append(distances.mean(axis=1))
    
    print(np.array(final_list))

        # additional data from others
        # additional_data_user_i_from_other = vstack([additional_data[:10*user_i], additional_data[10*(user_i+1):]]) 
        # # print(f"additional data from other {user_i}:",type(additional_data_user_i_from_other))
        # row_avg_user_additional_from_other = additional_data_user_i_from_other.mean(axis=0)
        # row_avg_user_additional_from_other = np.array(row_avg_user_additional_from_other).flatten()

        # # computing distance
        # l2_distance_same = euclidean(row_avg_user_i, row_avg_user_additional_from_i)
        # print(f"L2 original and additional same {user_i}: {l2_distance_same:.4f}")
        # row_avg_user_i = row_avg_user_i / np.sum(row_avg_user_i)

        # l2_distance_other = euclidean(row_avg_user_i, row_avg_user_additional_from_other)
        # print(f"L2 original and additional other {user_i}: {l2_distance_other:.4f}")
      


    
