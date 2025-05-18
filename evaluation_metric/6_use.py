import tensorflow_hub as hub
import numpy as np

# Load the Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Example corpus: Each document has 5 sentences
corpus = [
    [
        "The sun is shining brightly today.",
        "I love reading books in the afternoon.",
        "Machine learning is an exciting field.",
        "She enjoys hiking in the mountains.",
        "The coffee shop is crowded today."
    ],
    [
        "The dog is barking outside.",
        "He likes playing football with his friends.",
        "A new restaurant opened downtown.",
        "I am learning to play the piano.",
        "They traveled to Japan last summer."
    ]
]

# Flatten the corpus (since USE processes individual sentences)
flattened_sentences = [sentence for doc in corpus for sentence in doc]

# Compute embeddings
embeddings = model(flattened_sentences)

# Reshape back into document format (each document has 5 sentence embeddings)
doc_embeddings = np.array(embeddings).reshape(len(corpus), 5, -1)

# Print the shape of embeddings (num_documents, 5 sentences, 512 embedding size)
print("Embedding Shape:", doc_embeddings.shape)  # Expected: (num_documents, 5, 512)

