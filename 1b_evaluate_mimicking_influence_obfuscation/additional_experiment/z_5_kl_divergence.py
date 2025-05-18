import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from scipy.special import softmax

api ='gemini'
dataset ='speech'
root_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/additional_experiment/{dataset}/{api}/without_user_metadata/ppl_logs/'

# load writer for speech dataset
speakers = ['bush', 'obama', 'trump']

for speaker in speakers:
    # Load perplexity values from JSON files
    print(f"{speaker}")
    with open(root_path+ speaker+"_original.json", "r") as f:
        ppl_A = np.array(json.load(f))  # Convert to NumPy array

    with open(root_path+ speaker + "_mimicking_from_original.json", "r") as f:
        ppl_B = np.array(json.load(f))  # Convert to NumPy array

    with open(root_path+ speaker + "_mimicking_from_obfuscation.json", "r") as f:
        ppl_C = np.array(json.load(f))  # Convert to NumPy array
    
    with open(root_path+ speaker + "_obfuscation.json", "r") as f:
        ppl_D = np.array(json.load(f))  # Convert to NumPy array

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

    def compute_kl_divergence_softmax(ppl_A, ppl_B):
        """
        Compute KL divergence between two perplexity vectors using softmax normalization.

        Args:
            ppl_A (np.ndarray): Perplexity vector A.
            ppl_B (np.ndarray): Perplexity vector B.

        Returns:
            float: KL divergence D_KL(P || Q) where P = softmax(-ppl_A), Q = softmax(-ppl_B).
        """
        # Convert to numpy arrays in case inputs are lists
        ppl_A = np.array(ppl_A)
        ppl_B = np.array(ppl_B)

        # Softmax on negative perplexity (lower is better → higher probability)
        P = softmax(-ppl_A)
        Q = softmax(-ppl_B)

        # Compute KL divergence
        kl_div = entropy(P, Q)  # D_KL(P || Q)
        return kl_div


    # Compute KL Divergence
    kl_div = compute_kl_divergence(ppl_A, ppl_D)
    print(f"KL Divergence (Original; Obfuscation): {kl_div:.4f}")

    kl_div = compute_kl_divergence(ppl_A, ppl_B)
    print(f"KL Divergence (Original; Mimick from Original): {kl_div:.4f}")

    kl_div = compute_kl_divergence(ppl_A, ppl_C)
    print(f"KL Divergence (Original; Mimick from Obfuscation): {kl_div:.4f}")
