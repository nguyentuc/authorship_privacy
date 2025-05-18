import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Function to ask ChatGPT to synthesize a user profile
def generate_synthesize_dataset(avg, author_name, author_identification, sample_text, input_text):
    # Construct the prompt with variables
    prompt = f"You are an emulator designed to hide the writing style of a human author. You are given 5 sample writings from an author. The goal of this task is to conceal the author's writing style by carefully modifying lexical richness and diversity, sentence structure, punctuation patterns, special character usage, expressions and idioms, overall tone, emotion, mood, and any other distinguishing stylistic elements. Your task is to generate {avg}-word continuation that has writing style significantly different from the provided input text. Strive to make the rewritten text distinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 5 sample writings from {author_name}:\n{sample_text}\n\nThe input text is:\n{input_text}"
    # print(prompt)
    # exit()
    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
            )

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    return response.choices[0].message.content


# from the original dataset of the author, randomly sampling 5 for sample writing
# randomly 20% from the training set for obfuscation process 
def obfuscation_text(dataset_name):
    save_path =f"/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round5_step2/"

    if dataset_name == 'speech':
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")

            # read and select first 5 mimicking samples as groundtruth
            mimicking_text = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/mimicking/round5_step1/'+person+'.csv')
            sample_text = ''
            for idx, text in mimicking_text.head().iterrows():
                sample_text += text['Mimicking']+ '\n\n'

            # print(sample_text)
            # exit()
            # read author identification
            if person == 'trump':
                author_identification = "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            elif person == 'obama':
                author_identification = "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            else:
                author_identification = "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 

            # ramdonly select 20% for obfuscation
            # sample text that has bigger than 50 words
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))

            records = []
            i =0
            for ip_text in author_dataset:
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ' '.join(ip_text['text'].split(' ')[:15])   
                writing_sample = generate_synthesize_dataset(avg=60, author_name = person, author_identification=author_identification, sample_text=sample_text, input_text=input_text)
                writing_sample = writing_sample.replace('\n', '')
                print(f"Original text: {input_text}")
                print(f"Obfuscation text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Obfuscation'])
            df_record.to_csv(save_path+ person +'.csv', index=False)
    
    elif dataset_name=='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        for filename in os.listdir(root_path):
            if filename.endswith('.txt'):  # Check if the file is a .txt file
                file_path = os.path.join(root_path, filename)
                
                # Open and read the file
                with open(file_path, 'r') as file:
                    author_identification = file.read()

            # read and select first 5 mimicking samples as groundtruth
            mimicking_text = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/mimicking/round5_step1/'+filename.split('.')[0]+'.csv')
            sample_text = ''
            for idx, text in mimicking_text.head().iterrows():
                sample_text += text['Mimicking']+ '\n\n'

            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.1, random_state=42)

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Question']   
                writing_sample = generate_synthesize_dataset(avg=290, author_name = 'the author', author_identification=author_identification, sample_text=sample_text, input_text=input_text)
                writing_sample =writing_sample.replace('\n', '')
                print(f"Original text: {input_text}")
                print(f"Obfuscation text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Obfuscation'])
            df_record.to_csv(save_path+ filename.split('.')[0] +'.csv', index=False)


for dataset_name in ["speech",'quora']:
    obfuscation_text(dataset_name=dataset_name)