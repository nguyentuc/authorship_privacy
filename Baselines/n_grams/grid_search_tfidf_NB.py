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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

def load_dataset_k_user(data_path, k):
    train_texts, train_labels = [], []
    eval_texts, eval_labels = [], []
    test_texts, test_labels = [], []
    # train
    writing_files_train = os.listdir(data_path+'train/')
    random.seed(2024)
    writing_files_train = random.choices(writing_files_train, k=k)

    for writing in writing_files_train:
        writing_train = pd.read_csv(data_path +'train/'+ writing)
        writing_train = writing_train.sample(n=40, random_state=2024, replace=False)
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

    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'eval/'+ writing)
        for idx ,row  in writing_test.iterrows():
            eval_texts.append(row['Answer'])
            eval_labels.append(writing.split('.')[0])

        
    for writing in writing_files_train:
        writing_test = pd.read_csv(data_path +'test/'+ writing)
        for idx ,row  in writing_test.iterrows():
            test_texts.append(row['Answer'])
            test_labels.append(writing.split('.')[0])

    print(f"Train {len(train_texts)}, Valid {len(eval_texts)}, Test {len(test_texts)}")
    return train_texts, train_labels, eval_texts, eval_labels ,test_texts, test_labels

for k in [10, 50, 100, 150, 200]:
    train_data, train_labels, eval_data, eval_labels, test_data, test_labels = load_dataset_k_user('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/', k)

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



    # Convert the text data to TF-IDF features
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(train_data)
    X_eval = tfidf.transform(eval_data)
    X_test = tfidf.transform(test_data)


    best_alpha = None
    best_accuracy_eval = 0
    results = []
    # Train a Naive Bayes classifier (MultinomialNB)
    alphas = [0.001, 0.002, 0.005, 0.008, 0.01 , 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for alpha in alphas:
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(X_train, encoded_train_labels)

        # Evaluate on the training data
        train_preds = nb_classifier.predict(X_train)
        train_accuracy = accuracy_score(encoded_train_labels, train_preds)
        
        # Evaluate on the eval data
        eval_preds = nb_classifier.predict(X_eval)
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

    results_df = pd.DataFrame(results)
    # print("Results:\n", results_df)
    # print(f"Best alpha: {best_alpha} with eval accuracy: {best_accuracy_eval:.4f}")

    # train and evaluate model with best alpha on test set
    # print("Train and evaluate with best alpha")
    nb_classifier = MultinomialNB(alpha=best_alpha)
    nb_classifier.fit(X_train, encoded_train_labels)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(encoded_test_labels, y_pred)
    print(f"Multinomial NB: Accuracy with {k} users: {accuracy * 100:.2f}%")
    print(80*"=")
