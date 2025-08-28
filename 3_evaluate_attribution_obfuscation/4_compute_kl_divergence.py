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
    try:
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
        
        # Add small constant to avoid division by zero or log(0)
        epsilon = 1e-10
        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)
        
        # Compute KL divergence
        kl_div = entropy(P, Q)
        return kl_div
    except Exception as e:
        print(f"Error computing KL divergence: {e}")
        return np.nan


def analyze_perplexity_distributions(data_name, api, metadata_setting="with_user_metadata"):
    """
    Analyze perplexity distributions and compute KL divergence between original and obfuscated texts.
    
    Args:
        data_name: Name of the dataset ('speech' or 'quora')
        api: The LLM API used to generate texts
        metadata_setting: Whether to use results with user metadata
        
    Returns:
        Dictionary containing KL divergence values for each speaker
    """
    # Set paths
    root_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{data_name}/{api}/{metadata_setting}/ppl_logs/'
    
    # Define speakers
    speakers = ['bush', 'obama', 'trump']
    
    # Dictionary to store KL divergence results
    kl_divergence = {}
    perplexity_data = {}
    
    print(f"Analyzing perplexity distributions for {data_name} dataset with {api} API ({metadata_setting})")
    
    # Process each speaker
    for speaker in speakers:
        print(f"\nLoading perplexity data for {speaker}...")
        
        # Load perplexity values from JSON files
        try:
            with open(f"{root_path}{speaker}_original.json", "r") as f:
                ppl_original = np.array(json.load(f))
                
            with open(f"{root_path}{speaker}_obfuscation_from_correct.json", "r") as f:
                ppl_correct = np.array(json.load(f))
                
            with open(f"{root_path}{speaker}_obfuscation_from_incorrect.json", "r") as f:
                ppl_incorrect = np.array(json.load(f))
                
            # Store distributions for later visualization
            perplexity_data[speaker] = {
                'original': ppl_original,
                'obfuscation_correct': ppl_correct,
                'obfuscation_incorrect': ppl_incorrect
            }
            
            # Print basic statistics
            print(f"  Original: {len(ppl_original)} samples, mean={ppl_original.mean():.2f}, std={ppl_original.std():.2f}")
            print(f"  Obfuscation (correct): {len(ppl_correct)} samples, mean={ppl_correct.mean():.2f}, std={ppl_correct.std():.2f}")
            print(f"  Obfuscation (incorrect): {len(ppl_incorrect)} samples, mean={ppl_incorrect.mean():.2f}, std={ppl_incorrect.std():.2f}")
            
            # Compute KL divergence between original and obfuscated distributions
            kl_original_to_correct = compute_kl_divergence(ppl_original, ppl_correct)
            kl_original_to_incorrect = compute_kl_divergence(ppl_original, ppl_incorrect)
            
            # Calculate the average KL divergence
            avg_kl = (kl_original_to_correct + kl_original_to_incorrect) / 2
            kl_divergence[speaker] = avg_kl
            
            print(f"  KL divergence (original vs. correct obfuscation): {kl_original_to_correct:.4f}")
            print(f"  KL divergence (original vs. incorrect obfuscation): {kl_original_to_incorrect:.4f}")
            print(f"  Average KL divergence: {avg_kl:.4f}")
            
        except FileNotFoundError as e:
            print(f"  Error: {e}")
    
    # Calculate the overall average KL divergence
    valid_kl = [kl for kl in kl_divergence.values() if not np.isnan(kl)]
    if valid_kl:
        avg_kl = np.mean(valid_kl)
        print(f"\nAverage KL divergence across all speakers: {avg_kl:.4f}")
    else:
        avg_kl = np.nan
        print("\nNo valid KL divergence values to average.")
    
    # Create visualizations
    create_distribution_plots(perplexity_data, data_name, api, metadata_setting)
    
    return kl_divergence, avg_kl


def create_distribution_plots(perplexity_data, data_name, api, metadata_setting):
    """
    Create visualization of perplexity distributions.
    
    Args:
        perplexity_data: Dictionary containing perplexity distributions for each speaker
        data_name: Name of the dataset
        api: The LLM API used to generate texts
        metadata_setting: String indicating whether user metadata was used
    """
    if not perplexity_data:
        print("No data available for plotting.")
        return
        
    try:
        # Create figure for density plots
        plt.figure(figsize=(15, 10))
        
        for i, (speaker, distributions) in enumerate(perplexity_data.items()):
            plt.subplot(len(perplexity_data), 1, i+1)
            
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
        filename = f'perplexity_distributions_{data_name}_{api}_{metadata_setting}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved distribution plot to {filename}")
        
        # Create boxplot for better comparison of distributions
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        
        for speaker in perplexity_data:
            for dist_type, values in perplexity_data[speaker].items():
                boxplot_data.append(values)
                labels.append(f"{speaker.capitalize()} - {dist_type.replace('_', ' ').title()}")
        
        # Create boxplot
        plt.boxplot(boxplot_data, labels=labels, vert=False, showfliers=False)
        plt.title('Perplexity Distribution Comparison')
        plt.xlabel('Perplexity')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'perplexity_boxplot_{data_name}_{api}_{metadata_setting}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot to {filename}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")


# Main execution
if __name__ == "__main__":
    data_name = 'speech'
    api = 'gemini'
    metadata_setting = "with_user_metadata"
    
    # Analyze perplexity distributions and compute KL divergence
    kl_divergence, avg_kl = analyze_perplexity_distributions(data_name, api, metadata_setting)
    
    # Print summary
    print("\nKL Divergence Summary:")
    for speaker, kl in kl_divergence.items():
        print(f"  {speaker}: {kl:.4f}")
    print(f"\nOverall Average KL Divergence: {avg_kl:.4f}")
    
    # Save results to JSON
    results = {
        "individual_kl": kl_divergence,
        "average_kl": float(avg_kl)
    }
    
    output_file = f"kl_divergence_{data_name}_{api}_{metadata_setting}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")