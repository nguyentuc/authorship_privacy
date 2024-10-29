# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import random

def load_dataset_k_user(data_path, k):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    # train
    random.seed(2024)
    writing_files_train = os.listdir(data_path+'train/')
    writing_files_train = random.sample(writing_files_train, k)
    
    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train/'+ writing)
        for idx ,row  in writing_train.iterrows():
            train_texts.append(row['Answer'])
            train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with writing sample
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/benchmark_dataset/quora/synthesize_dataset_with_prompt_v2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_writing_sample'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with profile
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/benchmark_dataset/quora/synthesize_dataset_with_prompt_v2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_user_profile'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more with all
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/benchmark_dataset/quora/synthesize_dataset_with_prompt_v2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_full_information'])
    #         train_labels.append(writing.split('.')[0])

        
    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'test/'+ writing)
        for idx ,row  in writing_test.iterrows():
            test_texts.append(row['Answer'])
            test_labels.append(writing.split('.')[0])

    return train_texts, train_labels, test_texts, test_labels

for k in [10]:
    train_data, train_labels, test_data, test_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)
    print(len(set(train_labels)))
    print(len(set(test_labels))) #(some test have no writing)
    print(80*"=")

    labels_dictionary = {}
    idx = 0
    for label in train_labels:
        if label not in labels_dictionary:
            labels_dictionary[label] = idx
            idx+=1

    # encode new labels
    encoded_train_labels, encoded_test_labels = [],[]
    for label in train_labels:
        encoded_train_labels.append(labels_dictionary[label])

    for label in test_labels:
        encoded_test_labels.append(labels_dictionary[label])



    # Create N-grams using CountVectorizer (unigrams and bigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 4), analyzer='char')  # N-grams with unigrams, bigrams and trigrams

    # Fit and transform the training data
    X_train_ngrams = vectorizer.fit_transform(train_data)

    # Transform the test data (using the same vectorizer)
    X_test_ngrams = vectorizer.transform(test_data)

    # Train a Naive Bayes classifier (MultinomialNB)
    print("Evaluating Naive Bayes")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_ngrams, encoded_train_labels)
    # Predict on the test set
    y_pred = nb_classifier.predict(X_test_ngrams)
    accuracy = accuracy_score(encoded_test_labels, y_pred)
    print(f"Multinomial NB: Accuracy with {k} users: {accuracy * 100:.2f}%")

    # Logistic Regression
    # svd = TruncatedSVD(n_components=63, algorithm='randomized', random_state=0) # highly influence on the n_components
    # X_train_ngrams = svd.fit_transform(X_train_ngrams)
    # X_test_ngrams = svd.transform(X_test_ngrams)
    print("Evaluating Logistic Regression")
    clf = LogisticRegression(multi_class='multinomial', dual=False)
    clf.fit(X_train_ngrams, encoded_train_labels)

    y_pred = clf.predict(X_test_ngrams)
    accuracy = accuracy_score(encoded_test_labels, y_pred)
    print(f"Logistic Regression: Accuracy with {k} users :{accuracy * 100:.2f}%")
