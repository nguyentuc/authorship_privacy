import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Example corpus
documents = [
    "Machine learning is a fascinating field of artificial intelligence.",
    "Natural language processing enables machines to understand human language.",
    "Deep learning has improved the performance of AI models.",
    "Artificial intelligence is transforming industries worldwide.",
    "Topic modeling helps extract meaningful information from text."
]

# Tokenize and preprocess
texts = [word_tokenize(doc.lower()) for doc in documents]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model
num_topics = 2  # Number of topics to extract
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
