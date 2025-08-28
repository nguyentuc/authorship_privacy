import os
import json
import pandas as pd
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

def generate_synthesize_dataset(avg, author_identification, sample_text, input_text):
    """
    Generate text that mimics the writing style of an author based on sample texts.
    
    Args:
        avg: Target word count for the generated text
        author_identification: Information about the author
        sample_text: Sample writings from the author
        input_text: Starting text to continue from
        
    Returns:
        Generated text that mimics the author's style
    """
    # Construct the prompt
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
        # Make API request
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"


def mimicking_text(dataset_name):
    """
    Generate mimicked texts based on obfuscated samples in an iterative loop.
    This is round 5, step 1 in the iterative process.
    
    Args:
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    # Set up paths
    root_save = f"/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/mimicking/round5_step1/"
    obfuscation_path = f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round4_step2/'
    
    # Create output directory
    os.makedirs(root_save, exist_ok=True)
    
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
            
            # Load sample texts for mimicking (first 5 obfuscated samples from previous round)
            try:
                mimicking_data = pd.read_csv(obfuscation_file)
                if len(mimicking_data) < 5:
                    print(f"Warning: Not enough obfuscation samples for {person}")
                    sample_texts = mimicking_data['Obfuscation'].tolist()
                else:
                    sample_texts = mimicking_data.head()['Obfuscation'].tolist()
                
                sample_text = '\n\n'.join(sample_texts)
            except Exception as e:
                print(f"Error loading sample texts: {e}")
                continue
            
            # Prepare test data
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.1)))
            
            # Generate mimicked texts
            records = []
            
            for i, ip_text in enumerate(tqdm(test_dataset, desc=f"Generating mimicked texts for {person}")):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate mimicked text
                try:
                    writing_sample = generate_synthesize_dataset(
                        avg=60, 
                        author_identification=author_identification, 
                        sample_text=sample_text, 
                        input_text=input_text
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
                df_record.to_csv(os.path.join(root_save, f"{person}.csv"), index=False)
            else:
                print(f"No mimicked texts generated for {person}")
    
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
            
            # Load sample texts for mimicking (first 5 obfuscated samples from previous round)
            try:
                mimicking_data = pd.read_csv(obfuscation_file)
                if len(mimicking_data) < 5:
                    print(f"Warning: Not enough obfuscation samples for {author_id}")
                    sample_texts = mimicking_data['Obfuscation'].tolist()
                else:
                    sample_texts = mimicking_data.head()['Obfuscation'].tolist()
                
                sample_text = '\n\n'.join(sample_texts)
            except Exception as e:
                print(f"Error loading sample texts: {e}")
                continue
            
            # Load author's writings
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Warning: Writing file not found - {writing_file}")
                continue
                
            author_dataset = pd.read_csv(writing_file)
            test_dataset = author_dataset.sample(frac=0.1, random_state=42)
            
            # Generate mimicked texts
            records = []
            
            for i, (_, ip_text) in enumerate(tqdm(test_dataset.iterrows(), total=len(test_dataset), 
                                                 desc=f"Generating mimicked texts for {author_id}")):
                # Use question as input text
                input_text = ip_text['Question']
                
                # Generate mimicked text
                try:
                    writing_sample = generate_synthesize_dataset(
                        avg=290, 
                        author_identification=author_identification, 
                        sample_text=sample_text, 
                        input_text=input_text
                    )
                    
                    # Clean up the generated text
                    writing_sample = writing_sample.replace('\n', ' ')
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
                df_record.to_csv(os.path.join(root_save, f"{author_id}.csv"), index=False)
            else:
                print(f"No mimicked texts generated for {author_id}")


# Run the mimicking process for both datasets
if __name__ == "__main__":
    for dataset_name in ["speech", "quora"]:
        print(f"\n{'='*80}\nProcessing {dataset_name} dataset for round 5, step 1\n{'='*80}")
        mimicking_text(dataset_name=dataset_name)