import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde

api ='gemini'
dataset ='speech'
root_path = f'/media/volume/tucnv/Coding/AA/Loop_evaluation/obfuscation_ppl_logs.json'

def compute_kl_divergence(ppl_A, ppl_B, num_bins=50):
    # Estimate probability distributions using KDE
    kde_A = gaussian_kde(ppl_A)
    kde_B = gaussian_kde(ppl_B)

    # Create bins for probability estimation
    x = np.linspace(min(ppl_A.min(), ppl_B.min()), max(ppl_A.max(), ppl_B.max()), num_bins)

    # Evaluate probability densities
    P = kde_A(x)
    Q = kde_B(x)

    # Normalize distributions (convert to probability distributions)
    P /= P.sum()
    Q /= Q.sum()

    # Compute KL divergence (entropy function expects normalized probabilities)
    kl_div = entropy(P, Q)
    return kl_div
    
with open(root_path, "r") as file:
    ppls = json.load(file)
    
kl_div ={}
for data_corpus in ppls.keys():
    kl_div[data_corpus] ={}
    # ppls for each corpus
    ppl = ppls[data_corpus]
    
    # ppls for each round
    for round in ppl.keys():
        kl_div[data_corpus][round] ={}
        round_ppl = ppl[round]
        
        # for each author
        for author in round_ppl.keys():
            author_original_ppl = np.array(round_ppl[author]['original'])
            author_mimicking_ppl = np.array(round_ppl[author]['obfuscation'])
            # compute the kl divergence of original and mimicking
            kl_divergence = compute_kl_divergence(author_original_ppl, author_mimicking_ppl)
            kl_div[data_corpus][round][author] = kl_divergence
            
print(kl_div)

# compute average across round
avg_kl ={}
for data_corpus in kl_div.keys():
    avg_kl[data_corpus] ={}
    for round in kl_div[data_corpus].keys():
        avg_kl[data_corpus][round] = sum(kl_div[data_corpus][round].values()) / len(kl_div[data_corpus][round])

print("Avg KL")
print(avg_kl)
    
