import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde

data_name = 'speech'
api = 'gemini'
root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/with_user_metadata/ppl_logs/'

# load writer for speech dataset
speakers = ['bush', 'obama', 'trump']
kl ={}
for speaker in speakers:
    # dict for kl of speaker
    # Load perplexity values from JSON files
    print(f"{speaker}")
    with open(root_path+ speaker+"_original.json", "r") as f:
        ppl_A = np.array(json.load(f))  # Convert to NumPy array

    with open(root_path+ speaker + "_obfuscation_from_correct.json", "r") as f:
        ppl_B = np.array(json.load(f))  # Convert to NumPy array

    with open(root_path+ speaker + "_obfuscation_from_incorrect.json", "r") as f:
        ppl_C = np.array(json.load(f))  # Convert to NumPy array

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

    # Compute KL Divergence
    kl[speaker] = (compute_kl_divergence(ppl_A, ppl_B) + compute_kl_divergence(ppl_A, ppl_C))/2

print("All the KL mesurement:")
print(kl)
# compute avg of all the value 
print(f"Avg: {np.mean(list(kl.values()))}")

