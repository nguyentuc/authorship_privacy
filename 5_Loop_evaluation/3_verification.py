import os
import json
import pandas as pd
import backoff
from openai import OpenAI
from datasets import load_from_disk
from tqdm import tqdm

# API Constants
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Author information dictionary
AUTHOR_INFO = {
    'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
}

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def authorship_verification(author_name, author_identification, text_from_author, text_from_other, input_text):
    """
    Verify if the input text was written by the specified author using LLM analysis.
    
    Args:
        author_name: Name of the author to verify
        author_identification: Information about the author
        text_from_author: Sample writings from the author
        text_from_other: Sample writings from other authors
        input_text: Text to verify
        
    Returns:
        The LLM's verdict (yes or no)
    """
    # Construct the prompt
    prompt = (
        f"You are a judge designed to verify the attribution of a human-author written text. "
        f"You are given sample texts including 10 writings from the author and 10 writings from others. "
        f"Analyze the writing styles of the input text, disregarding the differences in topic and content. "
        f"Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, "
        f"affixes, quantities, humor, sarcasm, typographical errors, and misspellings. "
        f"Your task is to verify if the input text was written by {author_name}. "
        f"As output, exclusively return yes or no without any accompanying explanations or comments.\n\n"
        f"Here is some information about the author: {author_identification}.\n\n"
        f"The 10 sample writings from the author: \n{text_from_author}\n\n"
        f"The 10 sample writing from other: \n{text_from_other}\n\n"
        f"The input text is: \n{input_text}"
    )
    
    try:
        # Make API request
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="openai/o3-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
            seed=42,
            temperature=1.0,
            reasoning_effort="medium"
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {e}")
        raise


def verification(dataset_name):
    """
    Verify whether obfuscated texts are still attributable to their original authors.
    This is part of an iterative evaluation process.
    
    Args:
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    # Dictionary to store accuracy results
    accuracy_results = {}
    
    # Set paths
    obfuscation_path = f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round3_step2/'
    save_path = f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/verification/round3_step2/'
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Check if obfuscation file exists
            obfuscation_file = os.path.join(obfuscation_path, f"{person}.csv")
            if not os.path.exists(obfuscation_file):
                print(f"Warning: Obfuscation file not found - {obfuscation_file}")
                continue
                
            # Get author identification
            author_identification = AUTHOR_INFO.get(person, "Unknown author")
            
            # Get sample texts from the author for reference
            sample_author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            sample_author_dataset = sample_author_dataset.shuffle(seed=2025)
            sample_author_dataset = sample_author_dataset.select(range(10))
            
            sample_text = '\n\n'.join(text['text'] for text in sample_author_dataset)
            
            # Get sample texts from other authors
            other_dataset = dataset.filter(
                lambda example: example["style"] != person and len(example["text"].split()) > 50
            )['train']
            other_dataset = other_dataset.shuffle(seed=2025)
            other_samples = other_dataset.select(range(10))
            
            sample_text_from_other = '\n\n'.join(text['text'] for text in other_samples)
            
            # Load obfuscated texts for verification
            try:
                obfuscation_df = pd.read_csv(obfuscation_file)
                print(f"Loaded {len(obfuscation_df)} obfuscated texts for verification")
            except Exception as e:
                print(f"Error loading obfuscation file: {e}")
                continue
                
            # Perform verification
            records = []
            correct_count = 0
            
            for i, (_, ip_text) in enumerate(tqdm(obfuscation_df.iterrows(), total=len(obfuscation_df),
                                                 desc=f"Verifying texts for {person}")):
                input_text = ip_text['Obfuscation']
                
                # Get verification result
                try:
                    attribution_result = authorship_verification(
                        author_name=person,
                        author_identification=author_identification,
                        text_from_author=sample_text,
                        text_from_other=sample_text_from_other,
                        input_text=input_text
                    )
                    attribution_result = attribution_result.strip().lower()
                except Exception as e:
                    print(f"Error during verification, defaulting to 'no': {e}")
                    attribution_result = 'no'
                
                print(f"Text: {input_text[:100]}...")
                print(f"Authorship verification: {i+1}/{len(obfuscation_df)}: {attribution_result}")
                
                if attribution_result == 'yes':
                    correct_count += 1
                    
                records.append([input_text, attribution_result])
                print('-' * 80)
            
            # Calculate accuracy
            accuracy = correct_count / len(obfuscation_df) if len(obfuscation_df) > 0 else 0
            print(f"Accuracy for {person}: {accuracy:.4f}")
            accuracy_results[person] = accuracy
            
            # Save results
            df_record = pd.DataFrame(records, columns=['Input', 'Result'])
            df_record.to_csv(os.path.join(save_path, f"{person}.csv"), index=False)
        
        # Save overall results
        with open(os.path.join(save_path, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
        
        # Print summary
        print("\nAccuracy summary:")
        for person, acc in accuracy_results.items():
            print(f"  {person}: {acc:.4f}")
    
    elif dataset_name == 'quora':
        # Process Quora dataset
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Check if obfuscation file exists
            obfuscation_file = os.path.join(obfuscation_path, f"{author_id}.csv")
            if not os.path.exists(obfuscation_file):
                print(f"Warning: Obfuscation file not found - {obfuscation_file}")
                continue
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Get sample texts from the author for reference
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Warning: Writing file not found - {writing_file}")
                continue
                
            author_writings = pd.read_csv(writing_file)
            sample_writings = author_writings.sample(n=10, random_state=42)
            
            sample_text = '\n\n'.join(
                row['Question'] + ' ' + row['Answer'].replace('\n', '')
                for _, row in sample_writings.iterrows()
            )
            
            # Get sample texts from other authors
            other_authors = [f for f in os.listdir(profile_dir) if f != filename]
            all_writings = []
            
            for other_author in other_authors:
                other_id = other_author.split('.')[0]
                other_writing_file = os.path.join(writing_dir, f"{other_id}.csv")
                
                if os.path.exists(other_writing_file):
                    writings = pd.read_csv(other_writing_file)
                    all_writings.append(writings)
            
            if not all_writings:
                print(f"No writings found for other authors, skipping {author_id}")
                continue
                
            # Combine writings from other authors
            merged_other_writings = pd.concat(all_writings, ignore_index=True)
            negative_samples = merged_other_writings.sample(n=10, random_state=42)
            
            text_from_other = '\n\n'.join(
                row['Question'] + ' ' + row['Answer'].replace('\n', '')
                for _, row in negative_samples.iterrows()
            )
            
            # Load obfuscated texts for verification
            try:
                obfuscation_df = pd.read_csv(obfuscation_file)
                print(f"Loaded {len(obfuscation_df)} obfuscated texts for verification")
            except Exception as e:
                print(f"Error loading obfuscation file: {e}")
                continue
                
            # Perform verification
            records = []
            correct_count = 0
            
            for i, (_, ip_text) in enumerate(tqdm(obfuscation_df.iterrows(), total=len(obfuscation_df),
                                                 desc=f"Verifying texts for {author_id}")):
                input_text = ip_text['Obfuscation']
                
                # Get verification result
                try:
                    attribution_result = authorship_verification(
                        author_name="the author",
                        author_identification=author_identification,
                        text_from_author=sample_text,
                        text_from_other=text_from_other,
                        input_text=input_text
                    )
                    attribution_result = attribution_result.strip().lower()
                except Exception as e:
                    print(f"Error during verification, defaulting to 'no': {e}")
                    attribution_result = 'no'
                
                print(f"Text: {input_text[:100]}...")
                print(f"Authorship verification: {i+1}/{len(obfuscation_df)}: {attribution_result}")
                
                if attribution_result == 'yes':
                    correct_count += 1
                    
                records.append([input_text, attribution_result])
                print('-' * 80)
            
            # Calculate accuracy
            accuracy = correct_count / len(obfuscation_df) if len(obfuscation_df) > 0 else 0
            print(f"Accuracy for {author_id}: {accuracy:.4f}")
            accuracy_results[author_id] = accuracy
            
            # Save results
            df_record = pd.DataFrame(records, columns=['Input', 'Result'])
            df_record.to_csv(os.path.join(save_path, f"{author_id}.csv"), index=False)
        
        # Save overall results
        with open(os.path.join(save_path, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
        
        # Print summary
        print("\nAccuracy summary:")
        for author_id, acc in accuracy_results.items():
            print(f"  {author_id}: {acc:.4f}")


# Run verification for both datasets
if __name__ == "__main__":
    for name in ["speech", "quora"]:
        print(f"\n{'='*80}\nRunning verification for {name} dataset\n{'='*80}")
        verification(dataset_name=name)