import os
import json
import pandas as pd
import backoff
from openai import OpenAI
from datasets import load_from_disk

# Configuration
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OUTPUT_BASE_DIR = "/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/additional_experiment/"

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def authorship_verification(text_from_author, text_from_other, input_text):
    """
    Verify if the input text was written by the author based on sample texts.
    
    Args:
        text_from_author: Sample texts from the author
        text_from_other: Sample texts from other authors
        input_text: Text to verify
        
    Returns:
        "yes" or "no" verdict from the model
    """
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
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="openai/o3-mini",
            messages=[{
                "role": "developer",
                "content": [{
                    "type": "text",
                    "text": prompt,
                }]
            }],
            response_format={"type": "text"},
            seed=42,
            temperature=1.0,
            reasoning_effort="medium"
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {e}")
        raise


def verification(api, dataset_name):
    """
    Evaluate whether LLMs can attribute original texts to their authors based on mimicked samples.
    
    Args:
        api: The LLM API used for mimicking texts
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    # Set up paths
    save_folder = f'{OUTPUT_BASE_DIR}{dataset_name}/{api}/without/'
    os.makedirs(save_folder, exist_ok=True)
    
    if dataset_name == 'speech':
        # Define paths
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/with_user_metadata/'
        dataset_path = "/media/volume/tucnv/Coding/AA/Benchmark_generation/speech"
        
        # Load dataset
        dataset = load_from_disk(dataset_path)
        speakers = list(set(dataset['train']['style']))
        
        # Dictionary to store accuracy results
        accuracy_results = {}
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Read mimicked texts from CSV file
            mimicking_file = os.path.join(synthesize_dataset, 'mimicking_from_original', f'{person}.csv')
            if not os.path.exists(mimicking_file):
                print(f"Warning: File not found - {mimicking_file}")
                continue
                
            df = pd.read_csv(mimicking_file)
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Prepare sample texts from author (mimicked texts)
            sample_mimicking_text = '\n\n'.join(df_shuffled['Mimicking'][:10])
            
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
                        text_from_author=sample_mimicking_text, 
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
            df_record.to_csv(os.path.join(save_folder, f"{person}.csv"), index=False)
        
        # Print and save overall results
        print("\nOverall accuracy results:")
        for person, acc in accuracy_results.items():
            print(f"{person}: {acc:.4f}")
        
        # Save results to JSON
        with open(os.path.join(save_folder, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
    
    elif dataset_name == 'quora':
        # Define paths
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        # Dictionary to store accuracy results
        accuracy_results = {}
        
        for filename in os.listdir(profile_dir):
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Read mimicked texts from CSV file
            mimicking_file = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/quora/{api}/with_user_metadata/mimicking_from_original/{author_id}.csv'
            if not os.path.exists(mimicking_file):
                print(f"Warning: File not found - {mimicking_file}")
                continue
                
            df = pd.read_csv(mimicking_file)
            df_shuffled = df.sample(n=10, random_state=42).reset_index(drop=True)
            
            # Prepare sample texts from author (mimicked texts)
            sample_texts = []
            for _, text in df_shuffled.iterrows():
                sample_texts.append(text['Mimicking'].replace('\n', ''))
            sample_mimicking_text = '\n\n'.join(sample_texts)
            
            # Gather sample texts from other authors
            other_authors = [f for f in os.listdir(profile_dir) if f != filename]
            all_writings = []
            
            for other_author in other_authors:
                other_id = other_author.split('.')[0]
                writing_file = os.path.join(writing_dir, f"{other_id}.csv")
                
                if os.path.exists(writing_file):
                    writing = pd.read_csv(writing_file)
                    all_writings.append(writing)
            
            if not all_writings:
                print(f"No writings found for other authors")
                continue
                
            # Combine writings from other authors
            merged_other_writings = pd.concat(all_writings, ignore_index=True)
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
                        text_from_author=sample_mimicking_text, 
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
            df_record.to_csv(os.path.join(save_folder, f"{author_id}.csv"), index=False)
        
        # Print and save overall results
        print("\nOverall accuracy results:")
        for author_id, acc in accuracy_results.items():
            print(f"{author_id}: {acc:.4f}")
        
        # Save results to JSON
        with open(os.path.join(save_folder, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)


# Main execution
if __name__ == "__main__":
    # Run verification for specified API and dataset
    for api in ['gemini']:
        for dataset_name in ['speech']:
            print(f"\n{'='*80}\nRunning verification for {dataset_name} dataset using {api} API\n{'='*80}")
            verification(api=api, dataset_name=dataset_name)