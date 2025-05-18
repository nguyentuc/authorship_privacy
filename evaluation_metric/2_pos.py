import spacy
from collections import Counter
# Load the English model in spaCy
nlp = spacy.load('en_core_web_sm')

# Example set of documents
documents = [
    "Obama is the president.",
    "Trump was the former president of the United States.",
    "Bush served as president during the early 2000s."
]

# Initialize Counter
pos_counts = Counter()

# Process each document and compute POS tags
for doc in documents:
    spacy_doc = nlp(doc)  # Process the document with spaCy
    pos_counts.update(token.pos_ for token in spacy_doc)

print("Document:", pos_counts)
