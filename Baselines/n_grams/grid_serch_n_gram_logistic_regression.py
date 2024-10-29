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
        for idx ,row  in writing_train.iterrows():
            train_texts.append(row['Answer'])
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

    return train_texts, train_labels, eval_texts, eval_labels ,test_texts, test_labels

for k in [10, 29, 50, 100, 150, 200]:
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


    best_solver = None
    best_penalty = None
    best_c = None
    best_accuracy_eval = 0
    results = []

    print("Evaluating Logistic Regression")
    penalty =["l1", "l2", None]
    C = [100, 10, 1.0, 0.1, 0.01, 0.005, 0.0001]
    solvers = ["lbfgs", "newton-cg", "sag", "saga"]
    for c in C:
        for pen in penalty:
            for sol in solvers:
                if sol == 'lbfgs' and pen=='l1':
                    continue
                if sol == 'newton-cg' and pen=='l1':
                    continue
                if sol == 'newton-cholesky' and pen=='l1':
                    continue
                if sol == 'sag' and pen=='l1':
                    continue


                clf = LogisticRegression(multi_class='multinomial', dual=False, penalty=pen, C=c, solver=sol)
                clf.fit(X_train_ngrams, encoded_train_labels)

                # Evaluate on the training data
                train_preds = clf.predict(X_train_ngrams)
                train_accuracy = accuracy_score(encoded_train_labels, train_preds)
                
                # Evaluate on the eval data
                eval_preds = clf.predict(X_eval_ngrams)
                eval_accuracy = accuracy_score(encoded_eval_labels, eval_preds)

                # Store results
                results.append({
                    'penalty': pen,
                    'C': c,
                    'solvers': sol,
                    'train_accuracy': train_accuracy,
                    'eval_accuracy': eval_accuracy
                })

                # Update the best model if this one is better
                if eval_accuracy > best_accuracy_eval:
                    best_accuracy_eval = eval_accuracy
                    best_solver = sol
                    best_penalty = pen
                    best_c = c

    results_df = pd.DataFrame(results)
    print("Results:\n", results_df)
    print(f"Best sovler: {best_solver}, penalty:{best_penalty}, best c:{best_c} with eval accuracy: {best_accuracy_eval:.4f}")

    # train and evaluate model with best alpha on test set
    print("Train and evaluate with best hyper parameters:")
    clf = LogisticRegression(multi_class='multinomial', dual=False, penalty=best_penalty, C=best_c, solver=best_solver)
    clf.fit(X_train_ngrams, encoded_train_labels)
    y_pred = clf.predict(X_test_ngrams)
    accuracy = accuracy_score(encoded_test_labels, y_pred)
    print(f"Logistic Regression: Accuracy with {k} users: {accuracy * 100:.2f}%")
    print(80*"==")


# Logistic Regression
# svd = TruncatedSVD(n_components=63, algorithm='randomized', random_state=0) # highly influence on the n_components
# X_train_ngrams = svd.fit_transform(X_train_ngrams)
# X_test_ngrams = svd.transform(X_test_ngrams)
