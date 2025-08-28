import os
import json
import pandas as pd
import backoff
from openai import OpenAI
from datasets import load_from_disk
from tqdm import tqdm

# API keys and constants
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set environment variable for OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def authorship_verification(text_from_author, text_from_other, input_text, api):
    """
    Verify if the input text was written by the author based on sample texts.
    
    Args:
        text_from_author: Sample texts from the author
        text_from_other: Sample texts from other authors
        input_text: Text to verify
        api: The LLM API to use for verification
        
    Returns:
        "yes" or "no" verdict from the model
    """
    # Base prompt for all models
    prompt = (
        f"You are a judge designed to verify the attribution of a human-author written text. "
        f"You are given sample texts including 10 writings from the author and 10 writings from others. "
        f"Analyze the writing styles of the input text, disregarding the differences in topic and content. "
        f"Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, "
        f"affixes, quantities, humor, sarcasm, typographical errors, and misspellings. "
        f"Your task is to verify if the input text was written by the author. "
        f"As output, exclusively return yes or no without any accompanying explanations or comments.\n\n"
        f"The 10 sample writings from the author: \n{text_from_author}\n\n"
        f"The 10 sample writing from other: \n{text_from_other}\n\n"
        f"The input text is: \n{input_text}"
    )
    
    # Simplified prompt for GPT-4o-mini
    simplified_prompt = (
        f"You are given sample texts including 10 writings from the author and 10 writings from others. "
        f"Analyze the writing styles of the input text, disregarding the differences in topic and content. "
        f"Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, "
        f"affixes, quantities, humor, sarcasm, typographical errors, and misspellings. "
        f"Your task is to verify if the input text was written by the author. "
        f"As output, exclusively return yes or no without any accompanying explanations or comments.\n\n"
        f"The 10 sample writings from the author: \n{text_from_author}\n\n"
        f"The 10 sample writing from other: \n{text_from_other}\n\n"
        f"The input text is: \n{input_text}"
    )
    
    try:
        if api == 'o3-mini':
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=api,
                messages=[{
                    "role": "developer",
                    "content": [{"type": "text", "text": prompt}]
                }],
                response_format={"type": "text"},
                seed=42,
                temperature=1.0,
                reasoning_effort="medium"
            )
        elif api == 'gemini':
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": prompt}]
            )
        elif api == 'deepseek':
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
        else:  # Default to gpt-4o-mini
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                response_format={"type": "text"},
                seed=42,
                temperature=1.0,
                max_tokens=300,
                logprobs=True,
                messages=[
                    {"role": "system", "content": "You are a judge designed to verify the attribution of a human-author written text."},
                    {"role": "user", "content": simplified_prompt}
                ]
            )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {e}")
        raise


def verification(api, dataset_name):
    """
    Evaluate how the mix of original and obfuscated texts affects attribution accuracy.
    
    Args:
        api: The LLM API to use
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    if dataset_name == 'speech':
        # Define the ratios of original to obfuscated texts to test
        original_counts = [0, 2, 5, 8, 10]
        
        # Dictionary to store accuracy results for each ratio
        accuracy_results = {num: {} for num in original_counts}
        
        # Set paths
        obfuscation_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
        
        # Load dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        # Process each ratio of original to obfuscated texts
        for num_original in original_counts:
            print(f"\n{'='*60}\nTesting with {num_original} original and {10-num_original} obfuscated texts\n{'='*60}")
            
            # Process each speaker
            for person in speakers:
                print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
                
                # Load obfuscated texts
                obfuscation_file = os.path.join(obfuscation_path, 'obfuscation', f'{person}.csv')
                if not os.path.exists(obfuscation_file):
                    print(f"Warning: File not found - {obfuscation_file}")
                    continue
                    
                df = pd.read_csv(obfuscation_file)
                
                # Prepare sample texts from author (mix of obfuscated and original texts)
                obfuscated_samples = df['Obfuscation'][:10-num_original].tolist()
                
                # Add original samples if needed
                author_dataset = dataset.filter(
                    lambda example: example["style"] == person and len(example["text"].split()) > 50
                )['train']
                author_dataset = author_dataset.shuffle(seed=2024)
                author_dataset = author_dataset.shuffle(seed=2025)
                
                original_samples = [example['text'] for example in author_dataset.select(range(num_original))]
                
                # Combine samples
                sample_pos_text = '\n\n'.join(obfuscated_samples + original_samples)
                
                # Prepare test data (excluding samples used for training)
                remaining_indices = [i for i in range(len(author_dataset)) if i not in range(num_original)]
                filtered_dataset = author_dataset.select(remaining_indices)
                test_dataset = filtered_dataset.select(range(int(len(filtered_dataset) * 0.1)))
                
                # Get samples from other authors
                other_dataset = dataset.filter(
                    lambda example: example["style"] != person and len(example["text"].split()) > 50
                )['train']
                other_dataset = other_dataset.shuffle(seed=2025)
                other_samples = [example['text'] for example in other_dataset.select(range(10))]
                sample_text_from_other = '\n\n'.join(other_samples)
                
                # Run verification
                records = []
                correct_count = 0
                
                for i, ip_text in enumerate(tqdm(test_dataset, desc=f"Verifying {person}'s texts")):
                    input_text = ip_text['text']
                    
                    # Get verification result
                    try:
                        attribution_result = authorship_verification(
                            text_from_author=sample_pos_text,
                            text_from_other=sample_text_from_other,
                            input_text=input_text,
                            api=api
                        )
                        attribution_result = attribution_result.strip().lower()
                    except Exception as e:
                        print(f"Error during verification, defaulting to 'no': {e}")
                        attribution_result = 'no'
                    
                    print(f"Text: {input_text[:100]}...")
                    print(f"Authorship verification: {i+1}/{len(test_dataset)}: {attribution_result}")
                    
                    if attribution_result == 'yes':
                        correct_count += 1
                        
                    records.append([input_text, attribution_result])
                    print('-' * 80)
                
                # Calculate accuracy
                accuracy = correct_count / len(test_dataset)
                print(f"Accuracy for {person} with {num_original} original texts: {accuracy:.4f}")
                accuracy_results[num_original][person] = accuracy
                
                # Save results
                save_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/speech/{api}/without_user_metadata/attribution_from_{num_original}original_{10-num_original}obfuscation/'
                os.makedirs(save_path, exist_ok=True)
                
                df_record = pd.DataFrame(records, columns=['Input', 'Result'])
                df_record.to_csv(os.path.join(save_path, f"{person}.csv"), index=False)
        
        # Save overall results
        results_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/speech/{api}/without_user_metadata/'
        os.makedirs(results_path, exist_ok=True)
        
        with open(os.path.join(results_path, "ratio_accuracy_results.json"), "w") as f:
            json.dump(accuracy_results, f, indent=4)
        
        # Print summary
        print("\nAccuracy summary:")
        for num_original, speaker_results in accuracy_results.items():
            avg_accuracy = sum(speaker_results.values()) / len(speaker_results) if speaker_results else 0
            print(f"  {num_original} original texts: {avg_accuracy:.4f}")
    
    elif dataset_name == 'quora':
        # Define the ratios of original to obfuscated texts to test
        original_counts = [0, 5, 10]
        
        # Dictionary to store accuracy results for each ratio
        accuracy_results = {num: {} for num in original_counts}
        
        # Set paths
        obfuscation_path = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        # Process each ratio of original to obfuscated texts
        for num_original in original_counts:
            print(f"\n{'='*60}\nTesting with {num_original} original and {10-num_original} obfuscated texts\n{'='*60}")
            
            # Process each author
            for filename in os.listdir(profile_dir):
                if not filename.endswith('.txt'):
                    continue
                    
                author_id = filename.split('.')[0]
                print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
                
                # Load author's writings
                writing_file = os.path.join(writing_dir, f"{author_id}.csv")
                if not os.path.exists(writing_file):
                    print(f"Warning: File not found - {writing_file}")
                    continue
                    
                author_dataset = pd.read_csv(writing_file)
                
                # Select original samples
                original_samples = author_dataset.sample(n=num_original, random_state=42)
                original_texts = [
                    row['Question'] + ' ' + row['Answer'].replace('\n', '') 
                    for _, row in original_samples.iterrows()
                ]
                
                # Load obfuscated texts
                obfuscation_file = os.path.join(obfuscation_path, 'obfuscation', f'{author_id}.csv')
                if not os.path.exists(obfuscation_file):
                    print(f"Warning: File not found - {obfuscation_file}")
                    continue
                    
                df = pd.read_csv(obfuscation_file)
                obfuscated_texts = [
                    text.replace('\n', '') 
                    for text in df['Obfuscation'][:10-num_original]
                ]
                
                # Combine samples
                sample_pos_text = '\n\n'.join(original_texts + obfuscated_texts)
                
                # Get samples from other authors
                other_authors = [f for f in os.listdir(profile_dir) if f != filename]
                all_writings = []
                
                for other_author in other_authors:
                    other_id = other_author.split('.')[0]
                    other_writing_file = os.path.join(writing_dir, f"{other_id}.csv")
                    
                    if os.path.exists(other_writing_file):
                        writing = pd.read_csv(other_writing_file)
                        all_writings.append(writing)
                
                if not all_writings:
                    print(f"No writings found for other authors, skipping {author_id}")
                    continue
                    
                # Combine writings from other authors
                merged_other_writings = pd.concat(all_writings, ignore_index=True)
                merged_other_writings = merged_other_writings.sample(frac=1, random_state=42).reset_index(drop=True)
                negative_samples = merged_other_writings.sample(n=10, random_state=42)
                
                other_texts = [
                    row['Question'] + ' ' + row['Answer'].replace('\n', '') 
                    for _, row in negative_samples.iterrows()
                ]
                sample_text_from_other = '\n\n'.join(other_texts)
                
                # Prepare test data (excluding samples used for training)
                test_dataset = author_dataset.drop(original_samples.index)
                test_dataset = test_dataset.sample(frac=0.2, random_state=42)
                
                # Run verification
                records = []
                correct_count = 0
                
                for i, (_, ip_text) in enumerate(tqdm(test_dataset.iterrows(), total=len(test_dataset), 
                                                     desc=f"Verifying {author_id}'s texts")):
                    input_text = ip_text['Question'] + ' ' + ip_text['Answer'].replace('\n', '')
                    
                    # Get verification result
                    try:
                        attribution_result = authorship_verification(
                            text_from_author=sample_pos_text,
                            text_from_other=sample_text_from_other,
                            input_text=input_text,
                            api=api
                        )
                        attribution_result = attribution_result.strip().lower()
                    except Exception as e:
                        print(f"Error during verification, defaulting to 'no': {e}")
                        attribution_result = 'no'
                    
                    print(f"Text: {input_text[:100]}...")
                    print(f"Authorship verification: {i+1}/{len(test_dataset)}: {attribution_result}")
                    
                    if attribution_result == 'yes':
                        correct_count += 1
                        
                    records.append([input_text, attribution_result])
                    print('-' * 80)
                
                # Calculate accuracy
                accuracy = correct_count / len(test_dataset)
                print(f"Accuracy for {author_id} with {num_original} original texts: {accuracy:.4f}")
                accuracy_results[num_original][author_id] = accuracy
                
                # Save results
                save_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/quora/{api}/without_user_metadata/attribution_from_{num_original}original_{10-num_original}obfuscation/'
                os.makedirs(save_path, exist_ok=True)
                
                df_record = pd.DataFrame(records, columns=['Input', 'Result'])
                df_record.to_csv(os.path.join(save_path, f"{author_id}.csv"), index=False)
        
        # Save overall results
        results_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/quora/{api}/without_user_metadata/'
        os.makedirs(results_path, exist_ok=True)
        
        with open(os.path.join(results_path, "ratio_accuracy_results.json"), "w") as f:
            json.dump(accuracy_results, f, indent=4)
        
        # Print summary
        print("\nAccuracy summary:")
        for num_original, author_results in accuracy_results.items():
            avg_accuracy = sum(author_results.values()) / len(author_results) if author_results else 0
            print(f"  {num_original} original texts: {avg_accuracy:.4f}")


# Run verification
if __name__ == "__main__":
    verification(api='o3-mini', dataset_name='quora')