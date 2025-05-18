# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import random 
from sklearn import preprocessing
import numpy as np
import json

def load_dataset_k_user(data_path, k):
    train_texts, train_labels = [], []
    eval_texts, eval_labels = [], []
    test_texts, test_labels = [], []
    # train: sample user
    writing_files_train = os.listdir(data_path+'train_40_5_5/')
    np.random.seed(2024)
    writing_files_train = np.random.choice(writing_files_train, k, replace=False)

    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train_40_5_5/'+ writing)
        writing_train_sample = writing_train.sample(n=5, random_state=2024, replace=False) # choose only 5 to train, remaining will move to test
        for idx ,row  in writing_train_sample.iterrows():
            train_texts.append(row['Answer'])
            train_labels.append(writing.split('.')[0])

        # add remaining records to test
        remaining_df = writing_train.drop(writing_train_sample.index)
        for idx ,row  in remaining_df.iterrows():
            test_texts.append(row['Answer'])
            test_labels.append(writing.split('.')[0])

    # load 10 more original with writing sample
    # for writing in writing_files_train:
    #     try:
    #         synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/10_more_original_writing/'+ writing)
    #         for idx ,row  in synthesize_writing_only.iterrows():
    #             train_texts.append(row['Answer'])
    #             train_labels.append(writing.split('.')[0])
    #     except Exception as e:
    #         print(f"One user {writing} do not have enough to sample")
    #         pass

    # load 10 more synthesize with writing sample
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_writing_sample'])
    #         train_labels.append(writing.split('.')[0])

    # load 10 more synthesize with profile: remove:
    for writing in writing_files_train:
        # print('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_v4_prompt_remove_age_temp_0.2/')
        synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_v4_prompt_age_only_temp_0.2/'+ writing)[:10]
        for idx ,row  in synthesize_writing_only.iterrows():
            if row['Answer_with_user_profile'] == "I'm sorry, but I can't assisst with that." or row['Answer_with_user_profile'] == "I'm sorry, but I cannot assisst you with that." or row['Answer_with_user_profile']=='':
                print("Skipping: ", row['Answer_with_user_profile'])
                continue
            train_texts.append(row['Answer_with_user_profile'])
            train_labels.append(writing.split('.')[0])

    # load 10 more with all
    # for writing in writing_files_train:
    #     synthesize_writing_only = pd.read_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_with_prompt_temp_0.2/'+ writing)[:10]
    #     for idx ,row  in synthesize_writing_only.iterrows():
    #         train_texts.append(row['Answer_with_full_information'])
    #         train_labels.append(writing.split('.')[0])

    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'eval_40_5_5/'+ writing)
        for idx ,row  in writing_test.iterrows():
            eval_texts.append(row['Answer'])
            eval_labels.append(writing.split('.')[0])

        
    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'test_40_5_5/'+ writing)
        for idx ,row  in writing_test.iterrows():
            test_texts.append(row['Answer'])
            test_labels.append(writing.split('.')[0])

    print(f"{len(set(train_labels))} users : Train {len(train_texts)}, Valid {len(eval_texts)}, Test {len(test_texts)}")
    return train_texts, train_labels, eval_texts, eval_labels ,test_texts, test_labels

per_author_acc ={}

for k in [10, 50, 100, 150, 200]:
    train_data, train_labels, eval_data, eval_labels, test_data, test_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)
    # continue

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

    for label in eval_labels:
        encoded_eval_labels.append(labels_dictionary[label])

    for label in test_labels:
        encoded_test_labels.append(labels_dictionary[label])


    # Create N-grams using CountVectorizer (unigrams and bigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 4), analyzer='char')  # N-grams with unigrams, bigrams and trigrams

    # Fit and transform the training data
    X_train_ngrams = vectorizer.fit_transform(train_data)
    X_eval_ngrams = vectorizer.transform(eval_data)
    X_test_ngrams = vectorizer.transform(test_data)

    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_train_ngrams = max_abs_scaler.fit_transform(X_train_ngrams)
    X_eval_ngrams = max_abs_scaler.transform(X_eval_ngrams)
    X_test_ngrams = max_abs_scaler.transform(X_test_ngrams)

    best_alpha = None
    best_accuracy_eval = 0
    results = []
    # Train a Naive Bayes classifier (MultinomialNB)
    alphas = [0.001, 0.002, 0.005, 0.008, 0.01 , 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for alpha in alphas:
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(X_train_ngrams, encoded_train_labels)

        # Evaluate on the training data
        train_preds = nb_classifier.predict(X_train_ngrams)
        train_accuracy = accuracy_score(encoded_train_labels, train_preds)
        
        # Evaluate on the eval data
        eval_preds = nb_classifier.predict(X_eval_ngrams)
        eval_accuracy = accuracy_score(encoded_eval_labels, eval_preds)

        # Store results
        results.append({
            'alpha': alpha,
            'train_accuracy': train_accuracy,
            'eval_accuracy': eval_accuracy
        })

        # Update the best model if this one is better
        if eval_accuracy > best_accuracy_eval:
            best_accuracy_eval = eval_accuracy
            best_alpha = alpha

    # results_df = pd.DataFrame(results)
    # print("Results:\n", results_df)
    # print(f"Best alpha: {best_alpha} with eval accuracy: {best_accuracy_eval:.4f}")

    # train and evaluate model with best alpha on test set
    # print("Train and evaluate with best alpha")
    nb_classifier = MultinomialNB(alpha=best_alpha)
    nb_classifier.fit(X_train_ngrams, encoded_train_labels)
    y_pred = nb_classifier.predict(X_test_ngrams)
    accuracy = accuracy_score(encoded_test_labels, y_pred)
    print(f"Multinomial NB: Accuracy with {k} users: {accuracy * 100:.2f}%")

    # # y_true: actual labels, y_pred: predicted labels
    # cm = confusion_matrix(encoded_test_labels, y_pred)
    # # get correct classification for each user/ didive for total of writings each user.
    # per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    # # print("Per author accuracy:", per_class_accuracy)
    # per_author_acc[k] = per_class_accuracy.tolist()
    print(80*"=")

# Save dictionary to JSON file
# print("Saving results")
# with open("/media/volume/arkai-lab-data-private/Coding/AA/Baselines/n_grams/json_results/10_more_writing_synthesize_0.2.json", "w") as file:
#     json.dump(per_author_acc, file)

