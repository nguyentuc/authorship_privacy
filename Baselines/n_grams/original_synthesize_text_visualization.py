# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import random 
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_dataset_k_user(data_path, k):
    train_texts, train_labels = [], []
    # train
    writing_files_train = os.listdir(data_path+'train/')
    random.seed(2024)
    writing_files_train = random.choices(writing_files_train, k=k)

    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train/'+ writing)
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
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.8/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_writing_sample'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with profile
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.8/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_user_profile'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more with all
    for writing in writing_files_train:
        synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.8/'+ writing)[:10]
        for idx ,row  in synthesize_writing_only.iterrows():
            train_texts.append(row['Answer_with_full_information'])
            train_labels.append(writing.split('.')[0])
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

    # t-SNE for final 2D visualization
    for per in [10, 15, 25, 29, 30, 35]:
        for lr in [10, 20, 50, 80, 100]:
            for iter in [1500, 2000]:
                tsne = TSNE(n_components=2, random_state=2024, perplexity=per, learning_rate = lr ,max_iter=iter)
                X_tsne = tsne.fit_transform(X_train_ngrams.toarray())

                #plot
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=encoded_train_labels, cmap='viridis')
                plt.colorbar(scatter, label='Class')

                # Label the original data points on the first 5*k record
                for i in range(0, 5*k):
                    plt.text(X_tsne[i, 0], X_tsne[i, 1], encoded_train_labels[i], fontsize=8, ha='right', color='red')
                
                # label the additional dataset
                for i in range(5*k, X_tsne.shape[0]):
                    plt.text(X_tsne[i, 0], X_tsne[i, 1], encoded_train_labels[i], fontsize=8, ha='right', color='blue')

                plt.title(f"t-SNE visualization of n-gram embeddings on {k} user with original writing")
                plt.xlabel("t-SNE Dimension 1")
                plt.ylabel("t-SNE Dimension 2")
                plt.savefig(f"visualization_0.8/{k}_user_10_more_all_perplexity_{per}_learning_rate_{lr}_max_iter_{iter}.png", dpi=300)
