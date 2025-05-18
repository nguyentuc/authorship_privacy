# estimate topic distribution of all the documents group by each data corpus 
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


import os
import json
import pandas as pd
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub
import numpy as np
import torch.nn.functional as F
import tensorflow as tf


def LDA():
    stop_words = list(set(stopwords.words('english')))
    stop_words.append('and')
    stop_words.append('also')
    stop_words.append('the')
    stop_words.append('to')
    stop_words.append('we')
    stop_words.append('.')
    stop_words.append(',')
    stop_words.append('\'s')
    stop_words.append('--')
    stop_words.append('n\'t')
    stop_words.append('\'ve')
    stop_words.append('\'\'')
    stop_words.append('us')
    stop_words.append(';')
    stop_words.append('\'m')
    stop_words.append('must')
    stop_words.append('like')
    stop_words.append('every')
    stop_words.append('make')
    stop_words.append('would')
    stop_words.append('many')
    stop_words.append("'")
    stop_words.append('\'re')

    # load all the text need to compute
    # speakers = ['obama', 'bush', 'trump']
    # root_path = '/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/without_user_metadata/'
    # dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
    root_path = '/media/volume/tucnv/Coding/AA/Loop_evaluation/speech/with_user_metadata/mimicking/round5_step1/'
    all_writings =[]
    for speaker in os.listdir(root_path):
        print(f"Working on {speaker}")

        # original text
        # sample text that has bigger than 50 words  
        # author_dataset = dataset.filter(lambda example: example["style"] == speaker and len(example["text"].split()) > 50)['train']
        # author_dataset = author_dataset.shuffle(seed=2025)
        # author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.1)))
        
        record = pd.read_csv(root_path+speaker)
        for idx, rc in record.iterrows():
            all_writings.append(rc['Mimicking'])
    print(len(all_writings))
    
    ori_texts = [word_tokenize(doc.lower()) for doc in all_writings]
    texts =[]
    for doc_text in ori_texts:
        doc_t =[]
        for word in doc_text:
            if word not in stop_words:
                doc_t.append(word)
        texts.append(doc_t)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    num_topics = 10  # Number of topics to extract
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100, random_state=2025)
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(topic)
    

LDA()

# original
# (0, '0.010*"day" + 0.010*"election" + 0.009*"going" + 0.009*"people" + 0.008*"help" + 0.006*"votes" + 0.006*"working" + 0.005*"could" + 0.005*"got" + 0.005*"better"')
# (1, '0.009*"weapons" + 0.008*"tax" + 0.007*"best" + 0.007*"let" + 0.006*"people" + 0.006*"made" + 0.006*"could" + 0.006*"plan" + 0.005*"give" + 0.005*"think"')
# (2, '0.013*"people" + 0.010*"country" + 0.009*"time" + 0.007*"right" + 0.006*"look" + 0.006*"together" + 0.006*"one" + 0.005*"border" + 0.005*"want" + 0.005*"believe"')
# (3, '0.012*"iraq" + 0.008*"health" + 0.008*"costs" + 0.006*"people" + 0.006*"team" + 0.006*"looking" + 0.006*"war" + 0.006*"year" + 0.005*"care" + 0.005*"working"')
# (4, '0.018*"people" + 0.009*"jobs" + 0.009*"american" + 0.008*"time" + 0.007*"america" + 0.007*"states" + 0.006*"think" + 0.006*"right" + 0.006*"work" + 0.005*"put"')
# (5, '0.010*"new" + 0.009*"nation" + 0.008*"america" + 0.006*"american" + 0.005*"years" + 0.005*"right" + 0.005*"peace" + 0.005*"workers" + 0.005*"great" + 0.005*"drug"')
# (6, '0.010*"want" + 0.008*"people" + 0.007*"terrorists" + 0.006*"important" + 0.005*"?" + 0.005*"college" + 0.005*"enforcement" + 0.004*"asking" + 0.004*"terror" + 0.004*":"')
# (7, '0.010*"one" + 0.009*"security" + 0.008*"people" + 0.007*"country" + 0.006*"war" + 0.006*"life" + 0.006*"let" + 0.006*"never" + 0.006*"america" + 0.005*"american"')
# (8, '0.009*"going" + 0.006*"government" + 0.006*"economy" + 0.005*"world" + 0.005*"america" + 0.005*"afghanistan" + 0.005*"iraq" + 0.004*"getting" + 0.004*"history" + 0.004*"go"')
# (9, '0.012*"want" + 0.010*"going" + 0.010*"people" + 0.007*"americans" + 0.007*"think" + 0.006*"true" + 0.006*"test" + 0.006*"save" + 0.006*"health" + 0.006*"support"')

# round1_step1: mimicking
# (0, '0.016*"america" + 0.014*"nation" + 0.012*"people" + 0.011*"great" + 0.011*"believe" + 0.009*"world" + 0.008*"together" + 0.006*"continue" + 0.006*"better" + 0.006*"means"')
# (1, '0.009*"people" + 0.009*"states" + 0.008*"world" + 0.007*"new" + 0.007*"energy" + 0.007*"afghan" + 0.007*"united" + 0.007*"best" + 0.007*"take" + 0.006*"working"')
# (2, '0.023*"’" + 0.011*"weapons" + 0.009*"people" + 0.009*"country" + 0.009*"know" + 0.008*"america" + 0.005*"act" + 0.005*"got" + 0.005*"work" + 0.005*"tough"')
# (3, '0.008*"1st" + 0.007*"country" + 0.006*"good" + 0.006*"could" + 0.006*"china" + 0.006*"american" + 0.006*"always" + 0.006*"quality" + 0.006*"going" + 0.006*"people"')
# (4, '0.012*"nation" + 0.011*"iraq" + 0.011*"america" + 0.010*"united" + 0.009*"security" + 0.007*"’" + 0.007*"people" + 0.007*"states" + 0.007*"choose" + 0.007*"issue"')
# (5, '0.012*"people" + 0.010*"going" + 0.009*"new" + 0.008*"’" + 0.007*"americans" + 0.007*"great" + 0.005*"way" + 0.005*"thank" + 0.005*"american" + 0.005*"want"')
# (6, '0.022*"people" + 0.011*"going" + 0.009*"know" + 0.009*"right" + 0.009*"american" + 0.009*"one" + 0.008*"?" + 0.008*"get" + 0.008*"great" + 0.008*"’"')
# (7, '0.017*"people" + 0.014*"?" + 0.011*"going" + 0.010*"world" + 0.009*"american" + 0.008*"country" + 0.008*"families" + 0.007*"great" + 0.007*"challenges" + 0.007*"nation"')
# (8, '0.015*"want" + 0.013*"good" + 0.009*"working" + 0.009*"people" + 0.007*"continue" + 0.007*"america" + 0.007*"’" + 0.007*"get" + 0.006*"let" + 0.006*"need"')
# (9, '0.013*"america" + 0.012*"’" + 0.011*"going" + 0.009*"know" + 0.009*"day" + 0.008*"country" + 0.007*"people" + 0.007*"give" + 0.006*"future" + 0.006*"nation"')

# round1_step2: obfuscation
# (0, '0.018*"’" + 0.006*"energy" + 0.005*"progress" + 0.004*"remains" + 0.004*"across" + 0.004*"people" + 0.004*"iraq" + 0.004*"together" + 0.003*"let" + 0.003*"ensure"')
# (1, '0.101*"built" + 0.100*"probability" + 0.099*"plane" + 0.091*"0.0." + 0.050*"boeing" + 0.049*"airbus" + 0.007*"0.9." + 0.005*"’" + 0.002*"children" + 0.002*"life"')
# (2, '0.009*"’" + 0.005*"ensuring" + 0.005*"remain" + 0.004*"nation" + 0.004*"costs" + 0.004*"moving" + 0.004*"people" + 0.003*"yet" + 0.003*"accountability" + 0.003*"financial"')
# (3, '0.013*"’" + 0.005*"one" + 0.005*"ensuring" + 0.004*"future" + 0.004*"fostering" + 0.004*"ensure" + 0.004*"secure" + 0.004*"american" + 0.003*"communities" + 0.003*":"')
# (4, '0.009*"ensuring" + 0.009*"essential" + 0.008*"fostering" + 0.008*"sustainable" + 0.007*"future" + 0.007*"progress" + 0.007*"growth" + 0.007*"’" + 0.006*"efforts" + 0.006*"economic"')
# (5, '0.016*"’" + 0.005*"``" + 0.005*"one" + 0.004*"world" + 0.004*"unity" + 0.004*"life" + 0.004*"fostering" + 0.004*"resilience" + 0.003*"children" + 0.003*"wage"')
# (6, '0.007*"innovation" + 0.007*"’" + 0.006*"progress" + 0.005*"challenges" + 0.005*"yet" + 0.004*"ensuring" + 0.004*"fostering" + 0.004*"remains" + 0.004*"future" + 0.003*"essential"')
# (7, '0.006*"’" + 0.005*"commitment" + 0.004*"last" + 0.004*"crucial" + 0.004*"energy" + 0.004*"year" + 0.003*"world" + 0.003*"legal" + 0.003*"principles" + 0.003*"stability"')
# (8, '0.041*"’" + 0.012*"future" + 0.012*"progress" + 0.011*"let" + 0.010*"together" + 0.009*"challenges" + 0.009*"forward" + 0.007*"shared" + 0.007*"innovation" + 0.007*"resilience"')
# (9, '0.013*"’" + 0.006*"people" + 0.006*"time" + 0.005*"seemed" + 0.005*"yet" + 0.005*"dreams" + 0.004*"distant" + 0.004*"whispers" + 0.004*"future" + 0.004*"hope"')

#round2_step1: mimicking
# (0, '0.011*"people" + 0.010*"america" + 0.007*"time" + 0.007*"one" + 0.007*"great" + 0.006*"care" + 0.006*"future" + 0.006*"american" + 0.006*"could" + 0.006*"talking"')
# (1, '0.021*"’" + 0.014*"world" + 0.011*"going" + 0.010*"future" + 0.010*"let" + 0.007*"great" + 0.007*"build" + 0.007*"america" + 0.006*"job" + 0.006*"look"')
# (2, '0.010*"challenges" + 0.006*"america" + 0.006*"world" + 0.006*"new" + 0.006*"innovation" + 0.005*"good" + 0.004*"always" + 0.004*"moment" + 0.004*"americans" + 0.004*"embracing"')
# (3, '0.017*"people" + 0.011*"america" + 0.007*"let" + 0.007*"country" + 0.007*"opportunity" + 0.007*"world" + 0.007*"time" + 0.007*"essential" + 0.006*"’" + 0.006*"american"')
# (4, '0.008*"support" + 0.008*"people" + 0.007*"power" + 0.007*"’" + 0.007*"future" + 0.006*"let" + 0.006*"progress" + 0.006*"collective" + 0.006*"respect" + 0.005*"together"')
# (5, '0.008*"commitment" + 0.006*"’" + 0.005*"people" + 0.005*"principles" + 0.005*"trade" + 0.005*"essential" + 0.005*"future" + 0.005*"nation" + 0.004*"ensuring" + 0.004*"dedication"')
# (6, '0.011*"’" + 0.009*"work" + 0.009*"future" + 0.009*"people" + 0.008*"challenges" + 0.008*"let" + 0.008*"requires" + 0.006*"together" + 0.006*"vote" + 0.005*"commitment"')
# (7, '0.016*"’" + 0.010*"america" + 0.009*"together" + 0.007*"need" + 0.006*"nation" + 0.006*"challenges" + 0.006*"states" + 0.006*"total" + 0.005*"let" + 0.005*"open"')
# (8, '0.031*"’" + 0.026*"people" + 0.023*"going" + 0.009*"country" + 0.008*"great" + 0.007*"america" + 0.007*"let" + 0.006*"bad" + 0.006*"win" + 0.006*"things"')
# (9, '0.011*"’" + 0.008*"nation" + 0.007*"right" + 0.006*"progress" + 0.006*"values" + 0.006*"back" + 0.006*"believe" + 0.006*"going" + 0.006*"time" + 0.006*"security"')

# round2_step2: obfuscation
# (0, '0.021*"’" + 0.006*"let" + 0.006*"time" + 0.005*"together" + 0.004*"future" + 0.004*"essential" + 0.004*"ensure" + 0.004*"yet" + 0.004*"progress" + 0.004*"resilience"')
# (1, '0.026*"’" + 0.009*"progress" + 0.006*"forward" + 0.005*"let" + 0.005*"together" + 0.005*"path" + 0.004*"future" + 0.004*"ensuring" + 0.004*"yet" + 0.004*"test"')
# (2, '0.014*"’" + 0.013*"progress" + 0.012*"shared" + 0.012*"challenges" + 0.010*"future" + 0.010*"together" + 0.008*"unity" + 0.007*"resilience" + 0.007*"yet" + 0.007*"innovation"')
# (3, '0.013*"’" + 0.005*"innovation" + 0.005*"future" + 0.005*"could" + 0.004*"change" + 0.004*"time" + 0.004*"progress" + 0.004*"challenges" + 0.004*"resilience" + 0.004*"ensuring"')
# (4, '0.035*"’" + 0.007*"future" + 0.006*"let" + 0.005*"progress" + 0.004*"even" + 0.004*"fostering" + 0.004*"together" + 0.004*"challenges" + 0.004*"hope" + 0.004*"path"')
# (5, '0.005*"day" + 0.005*"time" + 0.004*"yet" + 0.004*"’" + 0.004*"relentless" + 0.004*"lies" + 0.003*"collective" + 0.003*"life" + 0.003*"fostering" + 0.003*"decisions"')
# (6, '0.011*"’" + 0.003*"offer" + 0.003*"energy" + 0.003*"greater" + 0.003*"yet" + 0.003*"dialogue" + 0.003*"fostering" + 0.003*"often" + 0.003*"today" + 0.003*"security"')
# (7, '0.006*"’" + 0.006*"people" + 0.005*"fostering" + 0.005*"progress" + 0.005*"solutions" + 0.005*"ensuring" + 0.005*"future" + 0.005*"innovation" + 0.005*"challenges" + 0.005*"efforts"')
# (8, '0.012*"’" + 0.007*"progress" + 0.006*"future" + 0.006*"collective" + 0.005*"resilience" + 0.005*"let" + 0.004*"together" + 0.004*"fostering" + 0.004*"solutions" + 0.004*"efforts"')
# (9, '0.011*"sustainable" + 0.010*"fostering" + 0.009*"growth" + 0.008*"future" + 0.008*"economic" + 0.008*"innovation" + 0.007*"ensuring" + 0.006*"essential" + 0.006*"’" + 0.006*"together"')

#round3_step1: mimicking
# (0, '0.017*"’" + 0.008*"jobs" + 0.008*"future" + 0.006*"american" + 0.006*"people" + 0.005*"look" + 0.005*"time" + 0.005*"let" + 0.005*"lot" + 0.005*"need"')
# (1, '0.011*"future" + 0.009*"nation" + 0.007*"together" + 0.006*"values" + 0.006*"people" + 0.006*"going" + 0.006*"build" + 0.006*"vital" + 0.005*"need" + 0.005*"american"')
# (2, '0.010*"commitment" + 0.008*"jobs" + 0.007*"right" + 0.006*"americans" + 0.006*"unwavering" + 0.006*"challenges" + 0.006*"economy" + 0.006*"fill" + 0.006*"good" + 0.006*"better"')
# (3, '0.012*"’" + 0.010*"world" + 0.007*"economy" + 0.006*"good" + 0.005*":" + 0.005*"fight" + 0.005*"prevail" + 0.005*"got" + 0.005*"forces" + 0.005*"iraq"')
# (4, '0.009*"together" + 0.008*"security" + 0.006*"great" + 0.005*"strategy" + 0.005*"even" + 0.005*"america" + 0.005*"need" + 0.005*"economic" + 0.005*"’" + 0.005*"made"')
# (5, '0.015*"people" + 0.013*"’" + 0.008*"future" + 0.008*"work" + 0.007*"america" + 0.006*"nation" + 0.006*"together" + 0.006*"values" + 0.006*"commitment" + 0.005*"american"')
# (6, '0.008*"get" + 0.006*"progress" + 0.006*"ahead" + 0.006*"future" + 0.006*"nation" + 0.004*"true" + 0.004*"americans" + 0.004*"shared" + 0.004*"tomorrow" + 0.004*"day"')
# (7, '0.017*"going" + 0.015*"people" + 0.009*"really" + 0.007*"see" + 0.006*"great" + 0.006*"know" + 0.005*"coming" + 0.005*"thing" + 0.005*"?" + 0.005*"election"')
# (8, '0.020*"’" + 0.008*"challenges" + 0.008*"together" + 0.007*"face" + 0.007*"world" + 0.007*"nation" + 0.007*"resolve" + 0.007*"forward" + 0.006*"future" + 0.006*"let"')
# (9, '0.014*"’" + 0.013*"always" + 0.010*"people" + 0.009*"believe" + 0.009*"right" + 0.009*"nation" + 0.008*"freedom" + 0.008*"country" + 0.008*"bad" + 0.008*"working"')

# round3_step2: obfuscation
# (0, '0.011*"’" + 0.006*"future" + 0.005*"energy" + 0.005*"collaboration" + 0.005*"fostering" + 0.005*"growth" + 0.004*"demands" + 0.004*"resilience" + 0.004*"sustainable" + 0.004*"economic"')
# (1, '0.019*"’" + 0.004*"ensuring" + 0.004*"forward" + 0.004*"innovation" + 0.004*"together" + 0.004*"progress" + 0.004*"demands" + 0.003*"stability" + 0.003*"clear" + 0.003*"economy"')
# (2, '0.015*"’" + 0.013*"together" + 0.013*"progress" + 0.009*"future" + 0.009*"let" + 0.008*"shared" + 0.008*"collaboration" + 0.008*"innovation" + 0.008*"challenges" + 0.007*"collective"')
# (3, '0.041*"’" + 0.007*"let" + 0.007*"progress" + 0.006*"yet" + 0.005*"together" + 0.005*"hope" + 0.004*"future" + 0.004*"resilience" + 0.004*"world" + 0.004*"time"')
# (4, '0.025*"’" + 0.010*"progress" + 0.008*"path" + 0.007*"let" + 0.007*"future" + 0.007*"yet" + 0.006*"forward" + 0.005*"together" + 0.005*"demands" + 0.005*"ahead"')
# (5, '0.010*"future" + 0.008*"’" + 0.007*"together" + 0.005*"hope" + 0.005*"ensuring" + 0.005*"efforts" + 0.005*"values" + 0.005*"unity" + 0.005*"resilience" + 0.004*"let"')
# (6, '0.006*"’" + 0.004*"quiet" + 0.004*"world" + 0.003*"shared" + 0.003*"becomes" + 0.003*"yet" + 0.003*"month" + 0.003*"america" + 0.003*"key" + 0.003*"let"')
# (7, '0.023*"’" + 0.007*"progress" + 0.006*"future" + 0.006*"challenges" + 0.006*"forward" + 0.005*"ensuring" + 0.005*"innovation" + 0.005*"resilience" + 0.005*"let" + 0.004*"time"')
# (8, '0.031*"’" + 0.006*"time" + 0.005*"global" + 0.005*"?" + 0.004*"need" + 0.004*"one" + 0.004*"north" + 0.003*"right" + 0.003*"let" + 0.003*"people"')
# (9, '0.024*"’" + 0.008*"future" + 0.007*"fairness" + 0.006*"ensuring" + 0.006*"ensure" + 0.005*"economic" + 0.005*"together" + 0.005*"everyone" + 0.004*"fostering" + 0.004*"collaboration"')

# round4_step1: mimicking
# (0, '0.014*"’" + 0.008*"get" + 0.007*"want" + 0.007*"future" + 0.006*"time" + 0.006*"let" + 0.006*"understand" + 0.004*"ensuring" + 0.004*"clear" + 0.004*"situation"')
# (1, '0.018*"’" + 0.016*"world" + 0.010*"people" + 0.008*"let" + 0.008*"american" + 0.008*"america" + 0.007*"challenges" + 0.007*"values" + 0.006*"stand" + 0.006*"together"')
# (2, '0.014*"people" + 0.009*"right" + 0.009*"’" + 0.009*"help" + 0.007*"innovation" + 0.007*"prosperity" + 0.007*"free" + 0.006*"thing" + 0.006*"economic" + 0.006*"families"')
# (3, '0.011*"going" + 0.010*"future" + 0.008*"’" + 0.008*"people" + 0.007*"need" + 0.007*"change" + 0.007*"country" + 0.005*"long" + 0.005*"better" + 0.005*"day"')
# (4, '0.016*"’" + 0.013*"people" + 0.009*"see" + 0.008*"things" + 0.006*"going" + 0.006*"something" + 0.006*"action" + 0.006*"stand" + 0.006*"disaster" + 0.006*"let"')
# (5, '0.021*"people" + 0.013*"bad" + 0.011*"great" + 0.011*"’" + 0.009*"nation" + 0.008*"america" + 0.007*"time" + 0.007*"see" + 0.007*"going" + 0.007*"country"')
# (6, '0.019*"’" + 0.009*"progress" + 0.008*"journey" + 0.007*"shaped" + 0.006*"nation" + 0.005*"spirit" + 0.005*"something" + 0.005*"always" + 0.005*"human" + 0.005*"going"')
# (7, '0.021*"’" + 0.008*"america" + 0.008*"together" + 0.008*"people" + 0.007*"future" + 0.007*"let" + 0.007*"world" + 0.007*"requires" + 0.006*"true" + 0.006*"get"')
# (8, '0.018*"’" + 0.010*"people" + 0.008*"support" + 0.007*"economy" + 0.006*"know" + 0.006*"work" + 0.006*"great" + 0.006*"world" + 0.005*"time" + 0.005*"done"')
# (9, '0.011*"world" + 0.009*"’" + 0.008*"america" + 0.008*"ahead" + 0.008*"future" + 0.007*"yet" + 0.007*"let" + 0.006*"resolve" + 0.006*"hope" + 0.006*"freedom"')

# round4_step2: obfuscation
# (0, '0.004*":" + 0.004*"guide" + 0.004*"future" + 0.004*"forward" + 0.003*"commitment" + 0.003*"essential" + 0.003*"wage" + 0.003*"next" + 0.003*"protect" + 0.003*"’"')
# (1, '0.012*"’" + 0.005*"path" + 0.004*"time" + 0.003*"something" + 0.003*"forward" + 0.003*"energy" + 0.003*"address" + 0.003*"$" + 0.003*"economic" + 0.003*"achieving"')
# (2, '0.036*"’" + 0.006*"let" + 0.006*"solutions" + 0.005*"forward" + 0.005*"?" + 0.005*"progress" + 0.005*"ahead" + 0.004*"time" + 0.003*"innovation" + 0.003*"change"')
# (3, '0.015*"’" + 0.005*"yet" + 0.005*"forward" + 0.004*"time" + 0.004*"small" + 0.004*"step" + 0.004*"care" + 0.004*"health" + 0.004*"forged" + 0.004*"energy"')
# (4, '0.011*"’" + 0.005*"together" + 0.005*"future" + 0.004*"world" + 0.004*"commitment" + 0.004*"yet" + 0.004*"challenges" + 0.004*"?" + 0.003*"let" + 0.003*"across"')
# (5, '0.018*"’" + 0.005*"*" + 0.004*"shared" + 0.004*"peace" + 0.004*"resilience" + 0.004*"path" + 0.004*"collaboration" + 0.003*"forward" + 0.003*"progress" + 0.003*"fostering"')
# (6, '0.024*"’" + 0.008*"future" + 0.006*"let" + 0.006*"forward" + 0.005*"progress" + 0.004*"path" + 0.004*"time" + 0.004*"challenges" + 0.004*"innovation" + 0.003*"keep"')
# (7, '0.023*"’" + 0.006*"progress" + 0.006*"together" + 0.005*"yet" + 0.005*"one" + 0.005*"let" + 0.005*"change" + 0.004*"time" + 0.003*"vision" + 0.003*"path"')
# (8, '0.017*"’" + 0.012*"progress" + 0.011*"together" + 0.008*"resilience" + 0.008*"challenges" + 0.007*"essential" + 0.007*"forward" + 0.007*"future" + 0.007*"collaboration" + 0.007*"ensuring"')
# (9, '0.040*"’" + 0.011*"future" + 0.010*"together" + 0.008*"let" + 0.008*"progress" + 0.006*"everyone" + 0.006*"fostering" + 0.005*"ensuring" + 0.005*"innovation" + 0.005*"build"')

# round5_step1: mimicking
# (0, '0.009*"together" + 0.006*"better" + 0.006*"great" + 0.005*"world" + 0.005*"path" + 0.005*"win" + 0.005*"’" + 0.004*"easy" + 0.004*"opportunity" + 0.004*"need"')
# (1, '0.028*"’" + 0.015*"let" + 0.008*"america" + 0.007*"?" + 0.007*"future" + 0.007*"country" + 0.007*"time" + 0.006*":" + 0.006*"get" + 0.006*"believe"')
# (2, '0.011*"get" + 0.011*"jobs" + 0.010*"let" + 0.008*"american" + 0.007*"’" + 0.007*"america" + 0.007*"?" + 0.006*"need" + 0.005*"means" + 0.005*"investing"')
# (3, '0.009*"american" + 0.009*"’" + 0.007*"world" + 0.006*"let" + 0.006*"unwavering" + 0.006*"ahead" + 0.005*"work" + 0.005*"opportunity" + 0.005*"people" + 0.005*"everyone"')
# (4, '0.038*"’" + 0.013*"let" + 0.011*"done" + 0.008*"people" + 0.008*"job" + 0.007*"forward" + 0.007*"time" + 0.006*"get" + 0.006*"citizens" + 0.006*"keep"')
# (5, '0.017*"’" + 0.012*"people" + 0.011*"america" + 0.010*"let" + 0.009*"get" + 0.009*"know" + 0.007*"country" + 0.007*"great" + 0.007*"right" + 0.006*"american"')
# (6, '0.009*"got" + 0.009*"ta" + 0.009*"room" + 0.006*"forward" + 0.006*"doubt" + 0.006*"ahead" + 0.006*"fight" + 0.006*"open" + 0.006*"freedom" + 0.006*"stay"')
# (7, '0.015*"people" + 0.009*"american" + 0.008*"work" + 0.007*"always" + 0.007*"america" + 0.007*"’" + 0.006*"time" + 0.006*"act" + 0.006*"future" + 0.006*"nation"')
# (8, '0.012*"’" + 0.009*"nation" + 0.009*"let" + 0.007*"best" + 0.007*"?" + 0.007*"got" + 0.006*"stay" + 0.006*"get" + 0.005*"folks" + 0.005*"sure"')
# (9, '0.027*"’" + 0.024*"going" + 0.020*"people" + 0.011*"know" + 0.009*"work" + 0.009*"really" + 0.008*"together" + 0.007*"believe" + 0.007*"want" + 0.006*"world"')

# round5_step2: obfuscation
# (0, '0.020*"’" + 0.006*"progress" + 0.006*"together" + 0.005*"shared" + 0.005*"resilience" + 0.005*"future" + 0.005*"challenges" + 0.005*"let" + 0.004*"fostering" + 0.004*"ensuring"')
# (1, '0.021*"’" + 0.006*"lost" + 0.005*"time" + 0.005*"one" + 0.005*"momentum" + 0.004*"path" + 0.004*"progress" + 0.004*"america" + 0.003*"?" + 0.003*"became"')
# (2, '0.021*"’" + 0.014*"future" + 0.011*"fostering" + 0.010*"innovation" + 0.010*"progress" + 0.007*"together" + 0.007*"embracing" + 0.006*"let" + 0.006*"resilience" + 0.006*"ensuring"')
# (3, '0.021*"’" + 0.009*"future" + 0.009*"progress" + 0.008*"ensuring" + 0.007*"let" + 0.006*"together" + 0.005*"challenges" + 0.005*"innovation" + 0.004*"collaboration" + 0.004*"vision"')
# (4, '0.006*"future" + 0.005*"’" + 0.005*"progress" + 0.005*"together" + 0.004*"yet" + 0.004*"forward" + 0.004*"resilience" + 0.004*"remains" + 0.003*"shared" + 0.003*"collective"')
# (5, '0.028*"’" + 0.008*"let" + 0.006*"progress" + 0.006*"together" + 0.005*"forward" + 0.005*"step" + 0.005*"commitment" + 0.005*"action" + 0.005*"ensure" + 0.004*"future"')
# (6, '0.025*"’" + 0.008*"together" + 0.007*"future" + 0.007*"progress" + 0.006*"let" + 0.005*"ensuring" + 0.005*"path" + 0.005*"unity" + 0.004*"shared" + 0.004*"commitment"')
# (7, '0.008*"’" + 0.006*"ensuring" + 0.006*"across" + 0.005*"future" + 0.005*"challenges" + 0.004*"resilience" + 0.004*"essential" + 0.004*"without" + 0.004*"forward" + 0.004*"together"')
# (8, '0.017*"’" + 0.006*"fostering" + 0.005*"let" + 0.005*"remains" + 0.004*"future" + 0.004*"progress" + 0.004*"approach" + 0.004*"financial" + 0.004*"essential" + 0.004*"together"')
# (9, '0.014*"’" + 0.008*"progress" + 0.006*"future" + 0.006*"shared" + 0.005*"forward" + 0.005*"together" + 0.005*"collaboration" + 0.005*"challenges" + 0.005*"yet" + 0.004*"innovation"')