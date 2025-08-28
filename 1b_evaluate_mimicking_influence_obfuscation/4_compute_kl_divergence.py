import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde

def compute_kl_divergence(ppl_A, ppl_B, num_bins=50):
    """
    Compute KL divergence between two perplexity distributions using KDE.
    
    Args:
        ppl_A: First distribution of perplexity values
        ppl_B: Second distribution of perplexity values
        num_bins: Number of bins for discretizing the distributions
        
    Returns:
        KL divergence value
    """
    # Handle edge cases
    if len(ppl_A) < 2 or len(ppl_B) < 2:
        print("Warning: Not enough data points for KDE estimation")
        return float('nan')
    
    # Estimate probability distributions using KDE
    try:
        kde_A = gaussian_kde(ppl_A)
        kde_B = gaussian_kde(ppl_B)
    except np.linalg.LinAlgError:
        print("Warning: KDE estimation failed, likely due to singular matrix")
        return float('nan')
    
    # Create bins for probability estimation
    x = np.linspace(min(ppl_A.min(), ppl_B.min()), max(ppl_A.max(), ppl_B.max()), num_bins)
    
    # Evaluate probability densities
    P = kde_A(x)
    Q = kde_B(x)
    
    # Normalize distributions (convert to probability distributions)
    P /= P.sum()
    Q /= Q.sum()
    
    # Small constant to avoid division by zero
    epsilon = 1e-10
    P = np.maximum(P, epsilon)
    Q = np.maximum(Q, epsilon)
    
    # Compute KL divergence
    kl_div = entropy(P, Q)
    
    return kl_div


def analyze_perplexity_distributions(data_name, api, with_metadata=True):
    """
    Analyze perplexity distributions and compute KL divergence between original and obfuscated texts.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        with_metadata: Whether to use results with user metadata
        
    Returns:
        Dictionary containing KL divergence values for each speaker
    """
    # Set up paths
    metadata_str = "with_user_metadata" if with_metadata else "without_user_metadata"
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_str}/ppl_logs/'
    
    # Define speakers for the speech dataset
    speakers = ['bush', 'obama', 'trump']
    
    # Dictionary to store KL divergence results
    kl_results = {}
    all_distributions = {}
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nAnalyzing perplexity distributions for {speaker}")
        
        # Load perplexity values from JSON files
        try:
            with open(root_path + speaker + "_original.json", "r") as f:
                ppl_original = np.array(json.load(f))
            
            with open(root_path + speaker + "_obfuscation_from_correct.json", "r") as f:
                ppl_obfus_correct = np.array(json.load(f))
            
            with open(root_path + speaker + "_obfuscation_from_incorrect.json", "r") as f:
                ppl_obfus_incorrect = np.array(json.load(f))
                
            # Store distributions for visualization
            all_distributions[speaker] = {
                'original': ppl_original,
                'obfuscation_correct': ppl_obfus_correct,
                'obfuscation_incorrect': ppl_obfus_incorrect
            }
            
            # Print basic statistics
            print(f"  Original: {len(ppl_original)} samples, mean={ppl_original.mean():.2f}, std={ppl_original.std():.2f}")
            print(f"  Obfuscation (correct): {len(ppl_obfus_correct)} samples, mean={ppl_obfus_correct.mean():.2f}, std={ppl_obfus_correct.std():.2f}")
            print(f"  Obfuscation (incorrect): {len(ppl_obfus_incorrect)} samples, mean={ppl_obfus_incorrect.mean():.2f}, std={ppl_obfus_incorrect.std():.2f}")
            
            # Compute KL divergence
            kl_orig_to_correct = compute_kl_divergence(ppl_original, ppl_obfus_correct)
            kl_orig_to_incorrect = compute_kl_divergence(ppl_original, ppl_obfus_incorrect)
            
            # Store average KL divergence
            kl_results[speaker] = (kl_orig_to_correct + kl_orig_to_incorrect) / 2
            
            print(f"  KL divergence (original vs. correct obfuscation): {kl_orig_to_correct:.4f}")
            print(f"  KL divergence (original vs. incorrect obfuscation): {kl_orig_to_incorrect:.4f}")
            print(f"  Average KL divergence: {kl_results[speaker]:.4f}")
            
        except FileNotFoundError as e:
            print(f"  Error: {e}")
    
    # Visualize the distributions
    create_distribution_plots(all_distributions, data_name, api, metadata_str)
    
    # Compute and print average KL divergence across all speakers
    valid_kl_values = [v for v in kl_results.values() if not np.isnan(v)]
    if valid_kl_values:
        avg_kl = np.mean(valid_kl_values)
        print(f"\nOverall KL divergence measurements:")
        for speaker, kl_value in kl_results.items():
            print(f"  {speaker}: {kl_value:.4f}")
        print(f"\nAverage KL divergence across all speakers: {avg_kl:.4f}")
    else:
        print("\nNo valid KL divergence values to average.")
    
    return kl_results


def create_distribution_plots(all_distributions, data_name, api, metadata_str):
    """
    Create visualization of perplexity distributions.
    
    Args:
        all_distributions: Dictionary containing perplexity distributions for each speaker
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_str: String indicating whether user metadata was used
    """
    try:
        plt.figure(figsize=(15, 10))
        
        for i, (speaker, distributions) in enumerate(all_distributions.items(), 1):
            plt.subplot(len(all_distributions), 1, i)
            
            # Plot each distribution using KDE
            sns.kdeplot(distributions['original'], label=f'Original ({len(distributions["original"])} samples)', alpha=0.7)
            sns.kdeplot(distributions['obfuscation_correct'], label=f'Obfuscation (correct) ({len(distributions["obfuscation_correct"])} samples)', alpha=0.7)
            sns.kdeplot(distributions['obfuscation_incorrect'], label=f'Obfuscation (incorrect) ({len(distributions["obfuscation_incorrect"])} samples)', alpha=0.7)
            
            plt.title(f'Perplexity Distribution for {speaker.capitalize()}')
            plt.xlabel('Perplexity')
            plt.ylabel('Density')
            plt.legend()
            
            # Add vertical lines for mean values
            plt.axvline(distributions['original'].mean(), color='blue', linestyle='--', alpha=0.5)
            plt.axvline(distributions['obfuscation_correct'].mean(), color='orange', linestyle='--', alpha=0.5)
            plt.axvline(distributions['obfuscation_incorrect'].mean(), color='green', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig(f'perplexity_distributions_{data_name}_{api}_{metadata_str}.png', dpi=300)
        print(f"\nSaved distribution plot to perplexity_distributions_{data_name}_{api}_{metadata_str}.png")
        
    except Exception as e:
        print(f"Error creating distribution plot: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'gemini'
    
    # Analyze with user metadata
    kl_results = analyze_perplexity_distributions(data_name, api, with_metadata=True)