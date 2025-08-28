import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Set up API keys (replace with your own)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"

# Set environment variable for OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def authorship_verification(author_name, author_identification, text_from_author, text_from_other, input_text, api):
    """
    Verify if the input text was written by the specified author using LLM analysis.
    
    Args:
        author_name: Name of the author to verify against
        author_identification: Information about the author
        text_from_author: Sample writings from the author
        text_from_other: Sample writings from other authors
        input_text: Text to verify
        api: Which LLM to use ('o3-mini', 'deepseek', 'gemini', or 'gpt-4o-mini')
        
    Returns:
        The LLM's verdict (yes or no)
    """
    # Base prompt for all models
    base_prompt = (
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
    
    # Simplified prompt for some models
    simplified_prompt = (
        f"You are given sample texts including 10 writings from the author and 10 writings from others. "
        f"Analyze the writing styles of the input text, disregarding the differences in topic and content. "
        f"Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, "
        f"affixes, quantities, humor, sarcasm, typographical errors, and misspellings. "
        f"Your task is to verify if the input text was written by {author_name}. "
        f"As output, exclusively return yes or no without any accompanying explanations or comments.\n\n"
        f"Here is some information about the author: {author_identification}.\n\n"
        f"The 10 sample writings from the author: \n{text_from_author}\n\n"
        f"The 10 sample writing from other: \n{text_from_other}\n\n"
        f"The input text is: \n{input_text}."
    )
    
    try:
        if api == 'o3-mini':
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=api,
                messages=[{
                    "role": "developer",
                    "content": [{"type": "text", "text": base_prompt}]
                }],
                response_format={"type": "text"},
                seed=42,
                temperature=1.0,
                reasoning_effort="medium"
            )
        elif api in ['deepseek', 'gemini']:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            model = "deepseek/deepseek-chat" if api == 'deepseek' else "google/gemini-2.0-flash-lite-001"
            prompt = base_prompt if api == 'deepseek' else simplified_prompt
            
            response = client.chat.completions.create(
                model=model,
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
                ],
            )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with API call: {e}")
        return "no"  # Default to "no" on error


def verification(api, dataset):
    """
    Run verification of text attribution using different combinations of sample texts.
    
    Args:
        api: The LLM API to use
        dataset: Dataset name ('speech' or 'quora')
    """
    if dataset == 'speech':
        # Set up paths
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/'
        output_dir = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/speech/{api}/with_user_metadata/attribution_from_original_10pos_10neg/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        dataset_data = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset_data['train']['style']))
        
        # Author information dictionary
        author_info = {
            'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
            'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
            'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 
        }
        
        # Track accuracy for each author
        accuracy_results = {}
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Select samples from mimicked texts
            mimicking_file = os.path.join(synthesize_dataset, 'mimicking_from_original', f'{person}.csv')
            if not os.path.exists(mimicking_file):
                print(f"Warning: File not found - {mimicking_file}")
                continue
                
            df = pd.read_csv(mimicking_file)
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Prepare mixed samples (0 mimicked + 10 original)
            mimicked_samples = df_shuffled['Mimicking'][:0].tolist()
            
            # Get original samples from the author
            author_dataset = dataset_data.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            original_samples = [example['text'] for example in author_dataset.select(range(10))]
            
            # Combine samples
            sample_texts_from_author = '\n\n'.join(mimicked_samples + original_samples)
            
            # Get samples from other authors
            other_dataset = dataset_data.filter(
                lambda example: example["style"] != person and len(example["text"].split()) > 50
            )['train']
            other_dataset = other_dataset.shuffle(seed=2025)
            other_samples = [example['text'] for example in other_dataset.select(range(10))]
            sample_texts_from_other = '\n\n'.join(other_samples)
            
            # Get author identification
            author_identification = author_info.get(person, "Unknown author")
            
            # Prepare test set (excluding samples used for training)
            remaining_indices = [i for i in range(len(author_dataset)) if i not in range(10)]
            filtered_dataset = author_dataset.select(remaining_indices)
            test_dataset = filtered_dataset.select(range(int(len(filtered_dataset) * 0.2)))
            
            # Run verification
            records = []
            correct_count = 0
            
            for i, ip_text in enumerate(test_dataset):
                input_text = ip_text['text']
                
                # Get verification result
                attribution_result = authorship_verification(
                    author_name=person,
                    author_identification=author_identification,
                    text_from_author=sample_texts_from_author,
                    text_from_other=sample_texts_from_other,
                    input_text=input_text,
                    api=api
                )
                
                # Clean up and process result
                attribution_result = attribution_result.strip().lower()
                
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
            df_record.to_csv(os.path.join(output_dir, f"{person}.csv"), index=False)
        
        # Save overall results
        print("\nOverall accuracy results:")
        for person, acc in accuracy_results.items():
            print(f"{person}: {acc:.4f}")
            
        with open(os.path.join(output_dir, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)
    
    elif dataset == 'quora':
        # Set up paths
        output_dir = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/quora/{api}/with_user_metada/attribution_from_0original_10mimicking_10neg/'
        os.makedirs(output_dir, exist_ok=True)
        
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        # Track accuracy for each author
        accuracy_results = {}
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load author's writings
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Warning: File not found - {writing_file}")
                continue
                
            author_dataset = pd.read_csv(writing_file)
            
            # Select original samples (0 samples)
            sample_text = author_dataset.sample(n=0, random_state=42)
            original_samples = []
            for _, text in sample_text.iterrows():
                original_samples.append(text['Question'] + ' ' + text['Answer'].replace('\n', ''))
            
            # Select mimicked samples
            mimicking_file = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/mimicking_from_original/{author_id}.csv'
            if not os.path.exists(mimicking_file):
                print(f"Warning: File not found - {mimicking_file}")
                continue
                
            df = pd.read_csv(mimicking_file)
            mimicked_samples = df.sample(n=10, random_state=42)['Mimicking'].apply(lambda x: x.replace('\n', '')).tolist()
            
            # Combine samples
            sample_texts_from_author = '\n\n'.join(original_samples + mimicked_samples)
            
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
                print(f"No writings found for other authors")
                continue
                
            merged_other_writings = pd.concat(all_writings, ignore_index=True)
            negative_samples = merged_other_writings.sample(n=10, random_state=42)
            
            other_texts = []
            for _, row in negative_samples.iterrows():
                other_texts.append(row['Question'] + ' ' + row['Answer'].replace('\n', ''))
            
            sample_texts_from_other = '\n\n'.join(other_texts)
            
            # Prepare test set (excluding samples used for reference)
            remaining_dataset = author_dataset.drop(sample_text.index)
            test_dataset = remaining_dataset.sample(frac=0.4, random_state=42)
            
            # Run verification
            records = []
            correct_count = 0
            
            for i, (_, ip_text) in enumerate(test_dataset.iterrows()):
                input_text = ip_text['Question'] + ' ' + ip_text['Answer'].replace('\n', '')
                
                # Get verification result
                attribution_result = authorship_verification(
                    author_name='the author',
                    author_identification=author_identification,
                    text_from_author=sample_texts_from_author,
                    text_from_other=sample_texts_from_other,
                    input_text=input_text,
                    api=api
                )
                
                # Clean up and process result
                attribution_result = attribution_result.strip().lower()
                
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
            df_record.to_csv(os.path.join(output_dir, f"{author_id}.csv"), index=False)
        
        # Save overall results
        print("\nOverall accuracy results:")
        for author_id, acc in accuracy_results.items():
            print(f"{author_id}: {acc:.4f}")
            
        with open(os.path.join(output_dir, "results.json"), "w") as json_file:
            json.dump(accuracy_results, json_file, indent=4)


# Run verification for the specified API and dataset
if __name__ == "__main__":
    verification(api='deepseek', dataset='quora')