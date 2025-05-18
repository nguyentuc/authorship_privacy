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