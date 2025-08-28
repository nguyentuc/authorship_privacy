import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# API keys and constants
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set environment variable for OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_obfuscation_text(avg, author_identification, sample_text, input_text, api):
    """
    Generate text that obfuscates the writing style of the original author.
    
    Args:
        avg: Target word count for the generated text
        author_identification: Information about the author
        sample_text: Sample writings from the author
        input_text: Starting text to continue from
        api: API model to use ('o3-mini', 'gemini', 'deepseek', or 'gpt-4o-mini')
        
    Returns:
        Generated text with obfuscated writing style
    """
    # Base prompt for all models
    base_prompt = (
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
        f"The 5 sample writings from the author:\n{sample_text}\n\n"
        f"The input text is:\n{input_text}"
    )
    
    try:
        if api == 'gemini':
            router_client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            completion = router_client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": base_prompt}]
            )
            return completion.choices[0].message.content
            
        elif api == 'o3-mini':
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
            return response.choices[0].message.content
            
        elif api == 'deepseek':
            router_client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            response = router_client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": base_prompt}]
            )
            return response.choices[0].message.content
            
        else:  # Default to gpt-4o-mini
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                response_format={"type": "text"},
                seed=42,
                temperature=1.0,
                max_tokens=400,
                logprobs=True,
                messages=[
                    {"role": "system", "content": "You are an emulator designed to hide the writing style of a human author."},
                    {"role": "user", "content": base_prompt}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating obfuscated text: {e}"


def text_obfuscation(api, dataset):
    """
    Generate obfuscated texts based on correctly attributed samples.
    
    Args:
        api: The LLM API to use
        dataset: Dataset name ('speech' or 'quora')
    """
    # Author identification information for speech dataset
    author_info = {
        'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 
    }
    
    if dataset == 'speech':
        # Set up paths
        attribution_path = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/speech/{api}/with_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/speech/{api}/with_user_metadata/obfuscation_from_correct_attribute/'
        os.makedirs(save_path, exist_ok=True)
        
        # Load dataset
        speech_data = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(speech_data['train']['style']))
        
        for person in speakers:
            print(f"\n{'='*40}\nWorking on: {person}\n{'='*40}")
            
            # Load correctly attributed texts
            attribution_file = os.path.join(attribution_path, f"{person}.csv")
            if not os.path.exists(attribution_file):
                print(f"Warning: File not found - {attribution_file}")
                continue
                
            df = pd.read_csv(attribution_file)
            df_filter = df[df['Result'] == 'yes'].sample(n=5, random_state=2025)
            
            if len(df_filter) < 5:
                print(f"Warning: Not enough correctly attributed samples for {person}")
                continue
                
            # Prepare sample texts that were correctly attributed
            sample_correct_attribution_text = '\n\n'.join(df_filter['Input'])
            
            # Get author identification
            author_identification = author_info.get(person, "Unknown author")
            
            # Get test data for obfuscation
            author_dataset = speech_data.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            # Generate obfuscated texts
            records = []
            
            for i, ip_text in enumerate(test_dataset):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate obfuscated text
                try:
                    writing_sample = generate_obfuscation_text(
                        avg=60, 
                        author_identification=author_identification, 
                        sample_text=sample_correct_attribution_text, 
                        input_text=input_text, 
                        api=api
                    )
                    
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
    
    elif dataset == 'quora':
        # Set up paths
        attribution_path = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/{dataset}/{api}/with_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/{dataset}/{api}/with_user_metadata/obfuscation_from_correct_attribute/'
        os.makedirs(save_path, exist_ok=True)
        
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\n{'='*40}\nWorking on: {author_id}\n{'='*40}")
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load correctly attributed texts
            attribution_file = os.path.join(attribution_path, f"{author_id}.csv")
            if not os.path.exists(attribution_file):
                print(f"Warning: File not found - {attribution_file}")
                continue
                
            df = pd.read_csv(attribution_file)
            df_filter = df[df['Result'] == 'yes'].sample(n=5, random_state=42)
            
            if len(df_filter) < 5:
                print(f"Warning: Not enough correctly attributed samples for {author_id}")
                continue
                
            # Prepare sample texts that were correctly attributed
            sample_attribution_text = '\n\n'.join(
                row['Input'].replace('\n', ' ') for _, row in df_filter.iterrows()
            )
            
            # Load author's writings for obfuscation
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Warning: File not found - {writing_file}")
                continue
                
            author_dataset = pd.read_csv(writing_file)
            test_dataset = author_dataset.sample(frac=0.2, random_state=42)
            
            # Generate obfuscated texts
            records = []
            
            for i, (_, ip_text) in enumerate(test_dataset.iterrows()):
                # Use question as input text
                input_text = ip_text['Question']
                
                # Generate obfuscated text
                try:
                    writing_sample = generate_obfuscation_text(
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


# Run the obfuscation process
if __name__ == "__main__":
    text_obfuscation(api='deepseek', dataset="quora")