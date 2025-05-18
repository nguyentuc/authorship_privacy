from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to compute Jaccard Similarity
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1 & set2) / len(set1 | set2)

# Example texts (original and paraphrased versions)
original_corpus = [
    "The cat sat on the mat.",
    "Machine learning is a subset of artificial intelligence."
]
paraphrased_corpus = [
    "A feline rested on the carpet.",
    "AI includes machine learning as one of its parts."
]

# Combine both corpora for consistent TF-IDF vectorization
all_texts = original_corpus + paraphrased_corpus

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split back into original and paraphrased matrices
original_vectors = tfidf_matrix[:len(original_corpus)]
paraphrased_vectors = tfidf_matrix[len(original_corpus):]

# Compute Cosine Similarity
similarity_scores = cosine_similarity(original_vectors, paraphrased_vectors)

# Compute Jaccard Similarity for each pair
jaccard_scores = [jaccard_similarity(original_corpus[i], paraphrased_corpus[i]) for i in range(len(original_corpus))]

# Compute Mean Similarity Scores
mean_cosine_similarity = np.mean(similarity_scores)
mean_jaccard_similarity = np.mean(jaccard_scores)

print("Mean Cosine Similarity Score:", mean_cosine_similarity)
print("Mean Jaccard Similarity Score:", mean_jaccard_similarity)