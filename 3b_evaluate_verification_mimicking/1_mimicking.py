import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
from tqdm import tqdm

# API keys and constants
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Author information dictionary
AUTHOR_INFO = {
    'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
    'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
}

def generate_text(avg, author_identification, sample_text, input_text, api):
    """
    Generate text that mimics the writing style of an author based on sample texts.
    
    Args:
        avg: Target word count for the generated text
        author_identification: Information about the author
        sample_text: Sample writings from the author
        input_text: Starting text to continue from
        api: API model to use ('o3-mini', 'gemini', 'deepseek', or 'gpt-4o-mini')
        
    Returns:
        Generated text that mimics the author's style
    """
    # Base prompt for all models
    prompt = (
        f"You are an emulator designed to replicate the writing style of a human author. "
        f"You are given 5 sample writings from the author. The goal of this task is to mimic "
        f"the author's writing style while paying meticulous attention to lexical richness and diversity, "
        f"sentence structure, punctuation style, special character style, expressions and idioms, "
        f"overall tone, emotion, and mood, or any other relevant aspect of writing style established "
        f"by the author. Your task is to generate a {avg}-word continuation that seamlessly blends "
        f"with the provided input text. Ensure that the continuation is indistinguishable from both "
        f"the input text and the 5 sample writings by the author. As output, exclusively return the "
        f"text completion without any accompanying explanations or comments.\n\n"
        f"Here is some information about the author: {author_identification}.\n\n"
        f"The 5 sample writings from the author:\n{sample_text}\n\n"
        f"The input text is:\n{input_text}"
    )
    
    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        if api == 'gemini':
            completion = client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": prompt}]
            )
        elif api == 'o3-mini':
            completion = client.chat.completions.create(
                model="openai/o3-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                seed=42,
                temperature=1.0,
                reasoning_effort="medium"
            )
        elif api == 'deepseek':
            completion = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
        else:  # Default to gpt-4o-mini
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                seed=42,
                temperature=1.0
            )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"


def text_mimicking(api, dataset_name):
    """
    Generate mimicked texts based on incorrectly attributed samples.
    
    Args:
        api: The LLM API to use
        dataset_name: Dataset name ('speech' or 'quora')
    """
    if dataset_name == 'speech':
        # Set up paths
        attribution_path = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/{dataset_name}/{api}/with_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/with_user_metadata/mimicking_from_incorrect_attribute/'
        os.makedirs(save_path, exist_ok=True)
        
        # Load dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Load incorrectly attributed texts
            attribution_file = os.path.join(attribution_path, f"{person}.csv")
            if not os.path.exists(attribution_file):
                print(f"Warning: File not found - {attribution_file}")
                continue
                
            df = pd.read_csv(attribution_file)
            df_filter = df[df['Result'] == 'no'].sample(n=5, random_state=2025)
            
            if len(df_filter) < 5:
                print(f"Warning: Not enough incorrectly attributed samples for {person}")
                # If there aren't enough samples, use replacement to get 5
                df_filter = df[df['Result'] == 'no'].sample(n=5, random_state=2025, replace=True)
            
            # Prepare sample texts that were incorrectly attributed
            sample_incorrect_attribution_text = '\n\n'.join(df_filter['Input'])
            
            # Get author identification
            author_identification = AUTHOR_INFO.get(person, "Unknown author")
            
            # Get test data for mimicking
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            # Generate mimicked texts
            records = []
            
            for i, ip_text in enumerate(tqdm(test_dataset, desc=f"Generating mimicked texts for {person}")):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate mimicked text
                try:
                    writing_sample = generate_text(
                        avg=60, 
                        author_identification=author_identification, 
                        sample_text=sample_incorrect_attribution_text, 
                        input_text=input_text, 
                        api=api
                    )
                    
                    combined_text = input_text + ' ' + writing_sample
                    
                    print(f"Original text: {input_text}")
                    print(f"Mimicking text {i+1}/{len(test_dataset)}")
                    print(combined_text)
                    
                    records.append([combined_text])
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                
                print('-' * 80)
            
            # Save results
            if records:
                print(f"Saving {len(records)} mimicked texts for {person}")
                df_record = pd.DataFrame(records, columns=['Mimicking'])
                df_record.to_csv(os.path.join(save_path, f"{person}.csv"), index=False)
            else:
                print(f"No mimicked texts generated for {person}")
    
    elif dataset_name == 'quora':
        # Set up paths
        attribution_path = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/{dataset_name}/{api}/with_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/with_user_metadata/mimicking_from_incorrect_attribute/'
        os.makedirs(save_path, exist_ok=True)
        
        profile_dir = f'/media/volume/tucnv/Coding/AA/Benchmark_generation/{dataset_name}/user_profile/'
        writing_dir = f'/media/volume/tucnv/Coding/AA/Benchmark_generation/{dataset_name}/writing/'
        
        # Process each author
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            output_file = os.path.join(save_path, f"{author_id}.csv")
            
            # Skip if already processed
            # if os.path.exists(output_file):
            #     print(f"Skipping {author_id} - already processed")
            #     continue
                
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load incorrectly attributed texts
            attribution_file = os.path.join(attribution_path, f"{author_id}.csv")
            if not os.path.exists(attribution_file):
                print(f"Warning: File not found - {attribution_file}")
                continue
                
            df = pd.read_csv(attribution_file)
            df_filter = df[df['Result'] == 'no'].sample(n=5, random_state=42, replace=True)
            
            if len(df_filter) < 1:
                print(f"Warning: No incorrectly attributed samples for {author_id}")
                continue
            
            # Prepare sample texts that were incorrectly attributed
            sample_texts = []
            for _, row in df_filter.iterrows():
                sample_texts.append(row['Input'].replace('\n', ' '))
            sample_attribution_text = '\n\n'.join(sample_texts)
            
            # Load author's writings for mimicking
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Warning: File not found - {writing_file}")
                continue
                
            author_dataset = pd.read_csv(writing_file)
            test_dataset = author_dataset.sample(frac=0.2, random_state=42)
            
            # Generate mimicked texts
            records = []
            
            for i, (_, ip_text) in enumerate(tqdm(test_dataset.iterrows(), total=len(test_dataset), 
                                                 desc=f"Generating mimicked texts for {author_id}")):
                # Use question as input text
                input_text = ip_text['Question']
                
                # Generate mimicked text
                try:
                    writing_sample = generate_text(
                        avg=290, 
                        author_identification=author_identification, 
                        sample_text=sample_attribution_text, 
                        input_text=input_text, 
                        api=api
                    )
                    
                    # Clean up the generated text
                    writing_sample = writing_sample.replace('\n', '')
                    combined_text = input_text + ' ' + writing_sample
                    
                    print(f"Original text: {input_text}")
                    print(f"Mimicking text {i+1}/{len(test_dataset)}")
                    print(f"{combined_text[:150]}...")
                    
                    records.append([combined_text])
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                
                print('-' * 80)
            
            # Save results
            if records:
                print(f"Saving {len(records)} mimicked texts for {author_id}")
                df_record = pd.DataFrame(records, columns=['Mimicking'])
                df_record.to_csv(output_file, index=False)
            else:
                print(f"No mimicked texts generated for {author_id}")


# Run the mimicking process for multiple APIs and datasets
if __name__ == "__main__":
    for api in ['gemini', 'o3-mini']:
        for dataset in ['speech', 'quora']:
            print(f"\n{'='*80}\nGenerating mimicked texts with {api} for {dataset} dataset\n{'='*80}")
            text_mimicking(api=api, dataset_name=dataset)