import os
import math
import pandas as pd
import numpy as np
import sacrebleu
from collections import Counter, defaultdict
from typing import List, Union
from datasets import load_from_disk
import nltk
from nltk.tokenize import word_tokenize

def tokenize(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace and converting to lowercase.
    
    Args:
        text: Input text string
        
    Returns:
        List of lowercase tokens
    """
    return text.lower().split()


def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: Size of n-grams to extract
        
    Returns:
        List of n-gram tuples
    """
    if n > len(tokens):
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_single(candidate: str, references: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score for a single candidate against multiple references.
    
    Args:
        candidate: The generated text
        references: List of reference texts
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    if not candidate.strip() or not references:
        return 0.0
    
    # Tokenize
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # Calculate brevity penalty
    candidate_len = len(candidate_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - candidate_len))
    
    if candidate_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / candidate_len) if candidate_len > 0 else 0.0
    
    # Calculate n-gram precisions
    log_precisions = []
    
    for n in range(1, max_n + 1):
        candidate_ngrams = get_ngrams(candidate_tokens, n)
        
        if not candidate_ngrams:
            log_precisions.append(-float('inf'))
            continue
        
        # Count candidate n-grams
        candidate_counts = Counter(candidate_ngrams)
        
        # For each reference, count n-grams and take maximum counts
        max_ref_counts = defaultdict(int)
        for ref_tokens in reference_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            ref_counts = Counter(ref_ngrams)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        
        # Calculate clipped counts
        clipped_counts = 0
        total_counts = 0
        
        for ngram, count in candidate_counts.items():
            clipped_counts += min(count, max_ref_counts[ngram])
            total_counts += count
        
        if total_counts == 0:
            precision = 0.0
        else:
            precision = clipped_counts / total_counts
        
        if precision == 0:
            log_precisions.append(-float('inf'))
        else:
            log_precisions.append(math.log(precision))
    
    # If all precisions are 0, return 0
    if all(p == -float('inf') for p in log_precisions):
        return 0.0
    
    # Calculate geometric mean of precisions
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    
    # Final BLEU score
    bleu = bp * math.exp(avg_log_precision)
    return bleu


def compute_bleu_corpus(candidates: List[str], references_list: List[List[str]], max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU score between lists of documents.
    
    Args:
        candidates: List of generated texts
        references_list: List of lists, where each inner list contains reference texts for corresponding candidate
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        Corpus-level BLEU score between 0 and 1
    """
    if len(candidates) != len(references_list):
        raise ValueError("Number of candidates must match number of reference lists")
    
    if not candidates or not references_list:
        return 0.0
    
    # Aggregate statistics across all documents
    total_candidate_len = 0
    total_ref_len = 0
    
    # For each n-gram order
    total_clipped_counts = [0] * max_n
    total_candidate_counts = [0] * max_n
    
    for candidate, references in zip(candidates, references_list):
        if not candidate.strip() or not references:
            continue
            
        candidate_tokens = tokenize(candidate)
        reference_tokens_list = [tokenize(ref) for ref in references]
        
        if not candidate_tokens:
            continue
        
        # Update lengths for brevity penalty
        candidate_len = len(candidate_tokens)
        ref_lens = [len(ref_tokens) for ref_tokens in reference_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - candidate_len))
        
        total_candidate_len += candidate_len
        total_ref_len += closest_ref_len
        
        # For each n-gram order
        for n in range(1, max_n + 1):
            candidate_ngrams = get_ngrams(candidate_tokens, n)
            
            if not candidate_ngrams:
                continue
            
            # Count candidate n-grams
            candidate_counts = Counter(candidate_ngrams)
            
            # For each reference, count n-grams and take maximum counts
            max_ref_counts = defaultdict(int)
            for ref_tokens in reference_tokens_list:
                ref_ngrams = get_ngrams(ref_tokens, n)
                ref_counts = Counter(ref_ngrams)
                for ngram, count in ref_counts.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
            
            # Calculate clipped counts for this document
            clipped_counts = 0
            total_counts = 0
            
            for ngram, count in candidate_counts.items():
                clipped_counts += min(count, max_ref_counts[ngram])
                total_counts += count
            
            total_clipped_counts[n-1] += clipped_counts
            total_candidate_counts[n-1] += total_counts
    
    # Calculate brevity penalty
    if total_candidate_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_candidate_len) if total_candidate_len > 0 else 0.0
    
    # Calculate n-gram precisions
    log_precisions = []
    for n in range(max_n):
        if total_candidate_counts[n] == 0:
            log_precisions.append(-float('inf'))
        else:
            precision = total_clipped_counts[n] / total_candidate_counts[n]
            if precision == 0:
                log_precisions.append(-float('inf'))
            else:
                log_precisions.append(math.log(precision))
    
    # If all precisions are 0, return 0
    if all(p == -float('inf') for p in log_precisions):
        return 0.0
    
    # Calculate geometric mean of precisions
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    
    # Final BLEU score
    bleu = bp * math.exp(avg_log_precision)
    return bleu


def compute_bleu_documents(candidates: List[str], references: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score between two lists of documents (1:1 correspondence).
    Each candidate is compared against its corresponding reference.
    
    Args:
        candidates: List of generated texts
        references: List of reference texts (same length as candidates)
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        Average BLEU score across all document pairs
    """
    if len(candidates) != len(references):
        raise ValueError("Candidates and references lists must have the same length")
    
    if not candidates or not references:
        return 0.0
    
    # Convert to format expected by corpus BLEU
    references_list = [[ref] for ref in references]
    return compute_bleu_corpus(candidates, references_list, max_n)


def compute_sacrebleu_score(candidates, references):
    """
    Compute BLEU score using sacrebleu library
    
    Args:
        candidates: List of candidate texts
        references: List of reference texts
        
    Returns:
        BLEU score object
    """
    return sacrebleu.corpus_bleu(candidates, [references])


def analyze_bleu_scores():
    """
    Analyze BLEU scores between original texts and their obfuscated/mimicked versions
    """
    # Configuration
    root_path = '/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/speech/deepseek/with_user_metadata/'
    dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
    speakers = ['obama', 'bush', 'trump']
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    # Results will be stored here
    results = {}
    
    for speaker in speakers:
        print(f"\n{'='*40}\nAnalyzing BLEU scores for {speaker.upper()}\n{'='*40}")
        results[speaker] = {}
        
        # Get original text
        author_dataset = dataset.filter(
            lambda example: example["style"] == speaker and len(example["text"].split()) > 50
        )['train']
        author_dataset = author_dataset.shuffle(seed=2024)
        author_dataset = author_dataset.shuffle(seed=2025)
        author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
        original_text = [example['text'] for example in author_dataset]
        
        # Load obfuscated text
        obfuscation_path = os.path.join(root_path, 'obfuscation_from_original', f'{speaker}.csv')
        if os.path.exists(obfuscation_path):
            df = pd.read_csv(obfuscation_path)
            obfuscation_text = df['Obfuscation'].tolist()
            
            # Ensure lists have the same length
            min_len = min(len(original_text), len(obfuscation_text))
            if min_len > 0:
                # Compute BLEU scores
                print(f"Computing BLEU scores between original and obfuscated texts...")
                
                # Our implementation
                custom_bleu = compute_bleu_documents(
                    original_text[:min_len], 
                    obfuscation_text[:min_len]
                )
                print(f"Custom document-level BLEU: {custom_bleu:.4f}")
                
                # SacreBLEU implementation
                try:
                    sacre_bleu = compute_sacrebleu_score(
                        original_text[:min_len], 
                        obfuscation_text[:min_len]
                    )
                    print(f"SacreBLEU score: {sacre_bleu.score:.4f}")
                    
                    # Store results
                    results[speaker]['custom_bleu'] = custom_bleu
                    results[speaker]['sacre_bleu'] = sacre_bleu.score
                    
                    # Detailed n-gram scores
                    print("\nN-gram precision scores (SacreBLEU):")
                    for i, p in enumerate(sacre_bleu.precisions):
                        print(f"  {i+1}-gram: {p:.2f}%")
                    
                    # Brevity penalty
                    print(f"Brevity penalty: {sacre_bleu.bp:.4f}")
                    print(f"Ratio of candidate to reference length: {sacre_bleu.sys_len}/{sacre_bleu.ref_len} = {sacre_bleu.sys_len/sacre_bleu.ref_len:.4f}")
                    
                except Exception as e:
                    print(f"Error computing SacreBLEU: {e}")
            else:
                print("No matching samples found for comparison")
        else:
            print(f"Warning: File not found - {obfuscation_path}")
    
    # Print summary
    print("\n\nSUMMARY OF BLEU SCORES:")
    print("=" * 40)
    for speaker, scores in results.items():
        print(f"{speaker.upper()}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.4f}")


if __name__ == "__main__":
    analyze_bleu_scores()