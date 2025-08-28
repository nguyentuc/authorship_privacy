import os
import json
import pandas as pd
import backoff
from openai import OpenAI
from datasets import load_from_disk

# Constants
OPENROUTER_API_KEY = "YOUR_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_BASE_DIR = "/media/volume/tucnv/Coding/AA/2b_evaluate_obfuscation_verification/additional_experiment/"
DATASET_BASE_DIR = "/media/volume/tucnv/Coding/AA/Benchmark_generation/"
OBFUSCATION_BASE_DIR = "/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/"

# Author identification information
AUTHOR_INFO = {
    'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
}


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def authorship_verification(author_name, author_identification, text_from_author, text_from_other, input_text):
    """
    Verify if the input text was written by the author based on sample texts.
    
    Args:
        author_name: Name of the author to verify
        author_identification: Information about the author
        text_from_author: Sample texts from the author
        text_from_other: Sample texts from other authors
        input_text: Text to verify
        
    Returns:
        "yes" or "no" verdict from the model
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
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="openai/o3-mini",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            response_format={"type": "text"},
            seed=42,
            temperature=1.0,
            reasoning_effort="medium"
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        raise


def verification(api, dataset_name):
    """
    Evaluate whether LLMs can attribute original texts to their authors based on obfuscated samples.
    
    Args:
        api: The LLM API to use
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    # Set up paths
    synthesize_dataset = f'{OBFUSCATION_BASE_DIR}{dataset_name}/{api}/with_user_metadata/'
    save_path = f'{OUTPUT_BASE_DIR}{dataset_name}/{api}/with_user_metadata/'
    os.makedirs(save_path, exist_ok=True)
    
    # Dictionary to store accuracy results
    accuracy_results = {}
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk(f"{DATASET_BASE_DIR}speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Load obfuscated texts
            obfuscation_file = os.path.join(synthesize_dataset, 'obfuscation', f'{person}.csv')
            if not os.path.exists(obfuscation_file):
                print(f"Warning: File not found - {obfuscation_file}")
                continue
                
            df = pd.read_csv(obfuscation_file)
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Prepare sample texts from author (obfuscated texts)
            sample_obfuscation_text = '\n\n'.join(df_shuffled['Obfuscation'][:10])
            
            # Get author identification
            author_identification = AUTHOR_INFO.get(person, "Unknown author")
            
            # Prepare test data (original texts from the author)
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            # Prepare sample texts from other authors
            other_dataset = dataset.filter(
                lambda example: example["style"] != person and len(example["text"].split()) > 50
            )['train']
            other_dataset = other_dataset.shuffle(seed=2025)
            sample_other = other_dataset.select(range(10))
            sample_text_from_other = '\n\n'.join(text['text'] for text in sample_other)
            
            # Perform verification
            records = []
            correct_count = 0
            
            for i, ip_text in enumerate(test_dataset):
                input_text = ip_text['text']
                
                # Verification process by LLMs
                try:
                    attribution_result = authorship_verification(
                        author_name=person,
                        author_identification=author_identification,
                        text_from_author=sample_obfuscation_text,
                        text_from_other=sample_text_from_other,
                        input_text=input_text
                    )
                    attribution_result = attribution_result.strip().lower()
                except Exception as e:
                    print(f"Error during verification: {e}")
                    attribution_result = 'no'
                
                print(f"Text: {input_text[:100]}...")
                print(f"Authorship verification: {i+1}/{len(test_dataset)}: {attribution_result}")
                
                if attribution_result == 'yes':
                    correct_count += 1
                    
                records.append([input_text, attribution_result])
                print('-' * 80)
            
            # Calculate accuracy
            accuracy = correct_count / len(test_dataset)
            print(f"Accuracy for {person}: {accuracy:.4f}")
            accuracy_results[person] = accuracy
            
            # Save results
            df_record = pd.DataFrame(records, columns=['Input', 'Result'])
            df_record.to_csv(os.path.join(save_path, f"{person}.csv"), index=False)
        
        # Save overall results to JSON
        results_file = os.path.join(save_path, "results.json")
        with open(results_file, "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
        print(f"Results saved to {results_file}")
    
    elif dataset_name == 'quora':
        # Define paths
        profile_dir = f'{DATASET_BASE_DIR}quora/user_profile/'
        writing_dir = f'{DATASET_BASE_DIR}quora/writing/'
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load obfuscated texts
            obfuscation_file = f'{OBFUSCATION_BASE_DIR}{dataset_name}/{api}/with_user_metadata/obfuscation/{author_id}.csv'
            if not os.path.exists(obfuscation_file):
                print(f"Warning: File not found - {obfuscation_file}")
                continue
                
            df = pd.read_csv(obfuscation_file)
            df_shuffled = df.sample(n=10, random_state=42).reset_index(drop=True)
            
            # Prepare sample texts from author (obfuscated texts)
            sample_texts = [text.replace('\n', '') for text in df_shuffled['Obfuscation']]
            sample_text = '\n\n'.join(sample_texts)
            
            # Gather sample texts from other authors
            other_authors = [f for f in os.listdir(profile_dir) if f != filename]
            all_writing = []
            
            for other_author in other_authors:
                other_id = other_author.split('.')[0]
                writing_file = os.path.join(writing_dir, f"{other_id}.csv")
                
                if os.path.exists(writing_file):
                    writing = pd.read_csv(writing_file)
                    all_writing.append(writing)
            
            if not all_writing:
                print(f"No writings found for other authors")
                continue
                
            # Combine writings from other authors
            merged_other_writings = pd.concat(all_writing, ignore_index=True)
            merged_other_writings = merged_other_writings.sample(frac=1, random_state=42).reset_index(drop=True)
            negative_samples = merged_other_writings.sample(n=10, random_state=42)
            
            # Prepare sample texts from other authors
            text_from_other = '\n\n'.join(
                row['Question'] + ' ' + row['Answer'].replace('\n', '') 
                for _, row in negative_samples.iterrows()
            )
            
            # Prepare test data (original texts from the author)
            author_writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(author_writing_file):
                print(f"Warning: File not found - {author_writing_file}")
                continue
                
            author_dataset = pd.read_csv(author_writing_file)
            test_dataset = author_dataset.sample(frac=0.2, random_state=42)
            
            # Perform verification
            records = []
            correct_count = 0
            
            for i, (_, ip_text) in enumerate(test_dataset.iterrows()):
                input_text = ip_text['Question'] + ' ' + ip_text['Answer'].replace('\n', '')
                
                # Verification process by LLMs
                try:
                    attribution_result = authorship_verification(
                        author_name='the author',
                        author_identification=author_identification,
                        text_from_author=sample_text,
                        text_from_other=text_from_other,
                        input_text=input_text
                    )
                    attribution_result = attribution_result.strip().lower()
                except Exception as e:
                    print(f"Error during verification: {e}")
                    attribution_result = 'no'
                
                print(f"Text: {input_text[:100]}...")
                print(f"Authorship verification: {i+1}/{len(test_dataset)}: {attribution_result}")
                
                if attribution_result == 'yes':
                    correct_count += 1
                    
                records.append([input_text, attribution_result])
                print('-' * 80)
            
            # Calculate accuracy
            accuracy = correct_count / len(test_dataset)
            print(f"Accuracy for {author_id}: {accuracy:.4f}")
            accuracy_results[author_id] = accuracy
            
            # Save results
            df_record = pd.DataFrame(records, columns=['Input', 'Result'])
            df_record.to_csv(os.path.join(save_path, f"{author_id}.csv"), index=False)
        
        # Save overall results to JSON
        results_file = os.path.join(save_path, "results.json")
        with open(results_file, "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    # Run verification for multiple APIs and datasets
    for api in ["o3-mini", "deepseek", "gemini"]:
        for name in ['speech', 'quora']:
            print(f"\n{'='*80}\nRunning verification for {name} dataset using {api} API\n{'='*80}")
            verification(api=api, dataset_name=name)