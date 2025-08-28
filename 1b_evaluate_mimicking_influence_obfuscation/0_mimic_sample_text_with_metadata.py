import os
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Set up API keys and constants
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set environment variable for OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

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
        f"The 5 sample writings from the author:\n{sample_text}\n\n"
        f"The input text is:\n{input_text}"
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
        elif api in ['gemini', 'deepseek']:
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
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
                max_tokens=400,
                logprobs=True,
                messages=[
                    {"role": "system", "content": "You are an emulator designed to hide the writing style of a human author."},
                    {"role": "user", "content": simplified_prompt}
                ],
            )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"


def generate_mimicking_text_for_fewshort_learning(api, dataset_name):
    """
    Generate mimicked text samples for few-shot learning experiments.
    
    Args:
        api: The LLM API to use
        dataset_name: Dataset type ('speech' or 'quora')
    """
    # Define output directory
    output_dir = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimciking_influence_obfuscation/{dataset_name}/{api}/with_user_metadata/micking_sample/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Author identification information for speech dataset
    author_info = {
        'trump': "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'obama': "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment",
        'bush': "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 
    }
    
    if dataset_name == 'speech':
        # Load speech dataset
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        speakers = list(set(dataset['train']['style']))
        
        for person in speakers:
            print(f"\nWorking on: {person}")
            
            # Filter dataset for current author with texts of sufficient length
            author_dataset = dataset.filter(
                lambda example: example["style"] == person and len(example["text"].split()) > 50
            )['train']
            
            # Select 5 samples for mimicking
            author_dataset = author_dataset.shuffle(seed=2024)
            sample_text_for_mimicking = author_dataset.select(range(5))
            
            # Select 5 different samples as ground truth for mimicking
            sample_ground_truth = author_dataset.shuffle(seed=42)
            sample_ground_truth = sample_ground_truth.select(range(5))
            sample_text_original = '\n\n'.join(text['text'] for text in sample_ground_truth)
            
            # Get author identification
            author_identification = author_info.get(person, "Unknown author")
            
            records = []
            for i, ip_text in enumerate(sample_text_for_mimicking):
                # Get the first 15 words for continuation
                input_text = ' '.join(ip_text['text'].split(' ')[:15])
                
                # Generate mimicked text
                writing_sample = generate_synthesize_dataset(
                    avg=60, 
                    author_identification=author_identification, 
                    sample_text=sample_text_original, 
                    input_text=input_text, 
                    api=api
                )
                
                combined_text = input_text + ' ' + writing_sample
                print(f"Sample {i+1}:\n{combined_text}")
                records.append([combined_text])
                print('-' * 80)
            
            # Save results
            print(f"Saving results for {person}")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(os.path.join(output_dir, f"{person}.csv"), index=False)
    
    elif dataset_name == 'quora':
        # Process Quora dataset
        profile_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        writing_dir = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'
        
        for filename in os.listdir(profile_dir):
            if not filename.endswith('.txt'):
                continue
                
            author_id = filename.split('.')[0]
            print(f"\nWorking on: {author_id}")
            
            # Read author profile
            with open(os.path.join(profile_dir, filename), 'r') as file:
                author_identification = file.read()
            
            # Load author's writings
            writing_file = os.path.join(writing_dir, f"{author_id}.csv")
            if not os.path.exists(writing_file):
                print(f"Writing file not found for {author_id}")
                continue
                
            author_dataset = pd.read_csv(writing_file)
            author_dataset = author_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Select 5 samples for mimicking
            sample_writing = author_dataset.sample(n=5, random_state=42)
            
            # Select 5 different samples as ground truth
            df_remaining = author_dataset.drop(sample_writing.index)
            sample_ground_truth = df_remaining.sample(n=5, random_state=42)
            
            # Prepare sample text from ground truth
            sample_text = '\n\n'.join(
                row['Question'] + ' ' + row['Answer'] 
                for _, row in sample_ground_truth.iterrows()
            )
            
            records = []
            for i, (_, ip_text) in enumerate(sample_writing.iterrows()):
                # Use the question as input text
                input_text = ip_text['Question']
                
                # Generate mimicked text
                writing_sample = generate_synthesize_dataset(
                    avg=290, 
                    author_identification=author_identification, 
                    sample_text=sample_text, 
                    input_text=input_text, 
                    api=api
                )
                
                # Clean up the generated text
                writing_sample = writing_sample.replace('\n', ' ')
                combined_text = input_text + ' ' + writing_sample
                
                print(f"Sample {i+1}:\n{combined_text[:150]}...")
                records.append([combined_text])
                print('-' * 80)
            
            # Save results
            print(f"Saving results for {author_id}")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(os.path.join(output_dir, f"{author_id}.csv"), index=False)


# Run the function to generate mimicked text samples
if __name__ == "__main__":
    generate_mimicking_text_for_fewshort_learning(api='o3-mini', dataset_name='quora')