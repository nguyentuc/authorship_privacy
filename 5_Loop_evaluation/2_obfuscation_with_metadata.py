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

def generate_synthesize_dataset(avg, author_name, author_identification, sample_text, input_text):
    """
    Generate text that obfuscates the writing style of the original author.
    
    Args:
        avg: Target word count for the generated text
        author_name: Name of the original author
        author_identification: Information about the author
        sample_text: Sample writings from the author
        input_text: Starting text to continue from
        
    Returns:
        Generated text with obfuscated writing style
    """
    # Construct the prompt
    prompt = (
        f"You are an emulator designed to hide the writing style of a human author. "
        f"You are given 5 sample writings from an author. The goal of this task is to conceal "
        f"the author's writing style by carefully modifying lexical richness and diversity, "
        f"sentence structure, punctuation patterns, special character usage, expressions and idioms, "
        f"overall tone, emotion, mood, and any other distinguishing stylistic elements. "
        f"Your task is to generate {avg}-word continuation that has writing style significantly "
        f"different from the provided input text. Strive to make the rewritten text distinguishable "
        f"from both the input text and the 5 sample writings by the author. As output, exclusively "
        f"return the text completion without any accompanying explanations or comments.\n\n"
        f"Here is some information about the author: {author_identification}.\n\n"
        f"The 5 sample writings from {author_name}:\n{sample_text}\n\n"
        f"The input text is:\n{input_text}"
    )
    
    try:
        # Make API request
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"


def obfuscation_text(dataset_name):
    """
    Generate obfuscated texts based on mimicked samples in an iterative loop.
    This is round 5, step 2 in the iterative process.
    
    Args:
        dataset_name: Dataset to use ('speech' or 'quora')
    """
    # Set up paths
    save_path = f"/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round5_step2/"
    mimicking_path = f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/mimicking/round5_step1/'
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Check if mimicking file exists
            mimicking_file = os.path.join(mimicking_path, f"{person}.csv")
            if not os.path.exists(mimicking_file):
                print(f"Warning: Mimicking file not found - {mimicking_file}")
                continue
                
            # Get author identification
            author_identification = AUTHOR_INFO.get(person, "Unknown author")
            
            # Load sample texts for obfuscation (first 5 mimicked samples from previous step)
            try:
                mimicking_data = pd.read_csv(mimicking_file)
                if len(mimicking_data) < 5:
                    print(f"Warning: Not enough mimicking samples for {person}")
                    sample_texts = mimicking_data['Mimicking'].tolist()
                else:
                    sample_texts = mimicking_data.head()['Mimicking'].tolist()
                
                sample_text = '\n\n'.join(sample_texts)
            except Exception as e:
                print(f"Error loading sample texts: {e}")
                continue
            
            # Prepare data for obfuscation
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            # Generate obfuscated texts
            records = []
            
            for i, ip_text in enumerate(tqdm(test_dataset, desc=f"Generating obfuscated texts for {person}")):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate obfuscated text
                try:
                    writing_sample = generate_synthesize_dataset(
                        avg=60, 
                        author_name=person, 
                        author_identification=author_identification, 
                        sample_text=sample_text, 
                        input_text=input_text
                    )
                    
                    # Clean up the generated text
                    writing_sample = writing_sample.replace('\n', '')
                    combined_text = input_text + ' ' + writing_sample
                    
                    print(f"Original text: {input_text}")
                    print(f"Obfuscation text {i+1}/{len(test_dataset)}")
                    print(combined_text)
                    
                    records.append([combined_text])
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                
                print('-' * 80)
            
            # Save results
            if records:
                print(f"Saving {len(records)} obfuscated texts for {person}")
                df_record = pd.DataFrame(records, columns=['Obfuscation'])
                df_record.to_csv(os.path.join(save_path, f"{person}.csv"), index=False)
            else:
                print(f"No obfuscated texts generated for {person}")
    
    elif dataset_name == 'quora':
        # Process Quora dataset
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Check if mimicking file exists
            mimicking_file = os.path.join(mimicking_path, f"{author_id}.csv")
            if not os.path.exists(mimicking_file):
                print(f"Warning: Mimicking file not found - {mimicking_file}")
                continue
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load sample texts for obfuscation (first 5 mimicked samples from previous step)
            try:
                mimicking_data = pd.read_csv(mimicking_file)
                if len(mimicking_data) < 5:
                    print(f"Warning: Not enough mimicking samples for {author_id}")
                    sample_texts = mimicking_data['Mimicking'].tolist()
                else:
                    sample_texts = mimicking_data.head()['Mimicking'].tolist()
                
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
            
            # Generate obfuscated texts
            records = []
            
            for i, (_, ip_text) in enumerate(tqdm(test_dataset.iterrows(), total=len(test_dataset), 
                                                 desc=f"Generating obfuscated texts for {author_id}")):
                # Use question as input text
                input_text = ip_text['Question']
                
                # Generate obfuscated text
                try:
                    writing_sample = generate_synthesize_dataset(
                        avg=290, 
                        author_name='the author', 
                        author_identification=author_identification, 
                        sample_text=sample_text, 
                        input_text=input_text
                    )
                    
                    # Clean up the generated text
                    writing_sample = writing_sample.replace('\n', '')
                    combined_text = input_text + ' ' + writing_sample
                    
                    print(f"Original text: {input_text}")
                    print(f"Obfuscation text {i+1}/{len(test_dataset)}")
                    print(f"{combined_text[:150]}...")
                    
                    records.append([combined_text])
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                
                print('-' * 80)
            
            # Save results
            if records:
                print(f"Saving {len(records)} obfuscated texts for {author_id}")
                df_record = pd.DataFrame(records, columns=['Obfuscation'])
                df_record.to_csv(os.path.join(save_path, f"{author_id}.csv"), index=False)
            else:
                print(f"No obfuscated texts generated for {author_id}")


# Run the obfuscation process for both datasets
if __name__ == "__main__":
    for dataset_name in ["speech", "quora"]:
        print(f"\n{'='*80}\nProcessing {dataset_name} dataset for round 5, step 2\n{'='*80}")
        obfuscation_text(dataset_name=dataset_name)