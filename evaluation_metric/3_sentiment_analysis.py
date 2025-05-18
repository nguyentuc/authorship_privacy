# https://huggingface.co/siebert/sentiment-roberta-large-english
from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Example set of documents
documents = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The movie was okay, but I expected more.",
    "Absolutely fantastic! Highly recommend.",
    "Not worth the money. Very disappointing."
]

# Make predictions
predictions = sentiment_pipeline(documents)

# Display results
for doc, pred in zip(documents, predictions):
    print(f"Text: {doc}\nSentiment: {pred['label']} (Confidence: {pred['score']:.4f})\n")
