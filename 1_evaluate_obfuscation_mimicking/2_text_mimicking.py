import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Set up the API key
os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"

def generate_synthesize_dataset(avg, author_identification, sample_text, input_text, api):
    """
    Generate text that mimics the writing style of an author.
    
    Args:
        avg: Target word count for the generated text
        author_identification: Information about the author
        sample_text: Sample writings from the author
        input_text: Starting text to continue from
        api: API model to use ('o3-mini', 'gemini', 'deepseek', or default to 'gpt-4o-mini')
        
    Returns:
        Generated text that mimics the author's style
    """
    # Base prompt for all models
    base_prompt = (
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
    
    # Simplified prompt for GPT-4o-mini
    simplified_prompt = (
        f"You are given 5 sample writings from the author. The goal of this task is to mimic "
        f"the author's writing style while paying meticulous attention to lexical richness and diversity, "
        f"sentence structure, punctuation style, special character style, expressions and idioms, "
        f"overall tone, emotion, and mood, or any other relevant aspect of writing style established "
        f"by the author. Your task is to generate a {avg}-word continuation that seamlessly blends "
        f"with the provided input text. Ensure that the continuation is indistinguishable from both "
        f"the input text and the 5 sample writings by the author. As output, exclusively return the "
        f"text completion without any accompanying explanations or comments.\n\n"
        f"Here is some information about the author: {author_identification}.\n\n"
        f"The 5 sample writings from an author:\n{sample_text}\n\n"
        f"The input text is:\n{input_text}"
    )
    
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
    elif api in ['gemini', 'deepseek']:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        model = "google/gemini-2.0-flash-lite-001" if api == 'gemini' else "deepseek/deepseek-chat"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": base_prompt}]
        )
    else:  # Default to gpt-4o-mini
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            response_format={"type": "text"},
            seed=42,
            temperature=1.0,
            max_tokens=500,
            logprobs=True,
            messages=[
                {"role": "system", "content": "You are an emulator designed to replicate the writing style of a human author."},
                {"role": "user", "content": simplified_prompt}
            ],
        )
    
    return response.choices[0].message.content


def mimicking_text(api, dataset):
    """
    Process a dataset to generate texts that mimic an author's writing style.
    
    Args:
        api: The LLM API to use
        dataset: Dataset type ('speech' or 'quora')
    """
    # Define paths
    root_save = f"/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/additional_experiment/{dataset}/{api}/with_user_metadata/"
    
    # Author identification information for speech dataset
    author_info = {
        'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 
    }
    
    if dataset == 'speech':
        # Set up paths
        synthesize_dataset = f"/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/"
        mimicking_output_dir = os.path.join(synthesize_dataset, "mimicking_from_original")
        os.makedirs(mimicking_output_dir, exist_ok=True)
        
        # Load speech dataset
        speech_data = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        authors = list(set(speech_data['train']['style']))
        
        for person in authors:
            print(f"Working on: {person}")
            
            # Read CSV and select first 5 obfuscations as a condition text for mimicking
            obfuscation_file = os.path.join(synthesize_dataset, "obfuscation", f"{person}.csv")
            if not os.path.exists(obfuscation_file):
                print(f"Obfuscation file not found for {person}")
                continue
                
            df = pd.read_csv(obfuscation_file)
            sample_text = '\n\n'.join(df['Obfuscation'].head().tolist())
            
            # Get author identification
            author_identification = author_info.get(person, "Unknown author")
            
            # Randomly select 20% for mimicking with 5 obfuscation text
            author_dataset = speech_data.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            test_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
            
            records = []
            for i, ip_text in enumerate(test_dataset):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate mimicked text
                writing_sample = generate_synthesize_dataset(
                    avg=60, 
                    author_identification=author_identification, 
                    sample_text=sample_text, 
                    input_text=input_text, 
                    api=api
                )
                
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i+1}/{len(test_dataset)}")
                print(input_text + ' ' + writing_sample)
                
                records.append([input_text + ' ' + writing_sample])
                print(80 * '-')
            
            # Save results
            print(f"Saving results for {person}")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(os.path.join(mimicking_output_dir, f"{person}.csv"), index=False)
    
    elif dataset == 'quora':
        # Set up paths
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        quora_output_dir = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/mimicking_from_original/'
        os.makedirs(quora_output_dir, exist_ok=True)
        
        for filename in os.listdir(profile_dir):
            if filename.endswith('.txt'):
                author_id = filename.split('.')[0]
                print(f"Working on: {author_id}")
                
                # Read author profile
                with open(os.path.join(profile_dir, filename), 'r') as file:
                    author_identification = file.read()
                
                # Load author's writings
                writing_file = os.path.join(writing_dir, f"{author_id}.csv")
                if not os.path.exists(writing_file):
                    print(f"Writing file not found for {author_id}")
                    continue
                    
                author_dataset = pd.read_csv(writing_file)
                
                # Get sample text from first 5 entries
                sample_text = '\n\n'.join(
                    row['Question'] + ' ' + row['Answer'] 
                    for _, row in author_dataset.head().iterrows()
                )
                
                # Select 40% for mimicking
                test_dataset = author_dataset.sample(frac=0.4, random_state=42)
                
                records = []
                for i, (_, ip_text) in enumerate(test_dataset.iterrows()):
                    # Use the question as input text
                    input_text = ip_text['Question']
                    
                    # Generate mimicked text with error handling
                    try:
                        writing_sample = generate_synthesize_dataset(
                            avg=290, 
                            author_identification=author_identification, 
                            sample_text=sample_text, 
                            input_text=input_text, 
                            api=api
                        )
                    except Exception as e:
                        print(f"Error generating text: {e}")
                        writing_sample = ip_text['Question'] + ' ' + ip_text['Answer']
                    
                    # Clean up the generated text
                    writing_sample = writing_sample.replace('\n', ' ')
                    
                    print(f"Original text: {input_text}")
                    print(f"Mimicking text {i+1}/{len(test_dataset)}")
                    print(input_text + ' ' + writing_sample)
                    
                    records.append([input_text + ' ' + writing_sample])
                    print(80 * '-')
                
                # Save results
                print(f"Saving results for {author_id}")
                df_record = pd.DataFrame(records, columns=['Obfuscation'])
                df_record.to_csv(os.path.join(quora_output_dir, f"{author_id}.csv"), index=False)


# Run the mimicking process for different APIs and datasets
if __name__ == "__main__":
    for api in ["4o-mini", "o3-mini", "gemini", "deepseek"]:
        for dataname in ["speech", "quora"]:
            print(f"\nProcessing {dataname} dataset with {api} API")
            mimicking_text(api=api, dataset=dataname)