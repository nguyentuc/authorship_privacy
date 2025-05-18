import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Function to ask ChatGPT to synthesize a user profile
def generate_synthesize_dataset(avg, author_identification, sample_text, input_text):
    prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
    # print(prompt)
    # exit()
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
    completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
    )
    return completion.choices[0].message.content

# from the obfuscation dataset of the author, randomly sampling 5 for sample writing.
# randomly 10% from the training set for mimicking process. 
# for each sample, call API for mimicking.
def mimicking_text(dataset_name):
    root_save =f"/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/mimicking/round5_step1/"
    if dataset_name=='speech':
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")

            # author_identification 
            if person == 'trump':
                author_identification = "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            elif person == 'obama':
                author_identification = "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            else:
                author_identification = "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 

            # sample text for mimicking
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.1)))

            # read and select first 5 obfuscation samples as groundtruth
            mimicking_text = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round4_step2/'+person+'.csv')
            sample_text = ''
            for idx, text in mimicking_text.head().iterrows():
                sample_text += text['Obfuscation']+ '\n\n'

            records = []
            i =0
            for ip_text in author_dataset:
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ' '.join(ip_text['text'].split(' ')[:15])   

                writing_sample = generate_synthesize_dataset(avg=60, author_identification=author_identification, sample_text=sample_text, input_text=input_text)
                
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(root_save +person +'.csv', index=False)
    
    elif dataset_name =='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        for filename in os.listdir(root_path):
            if filename.endswith('.txt'):  # Check if the file is a .txt file
                file_path = os.path.join(root_path, filename)
                
                # Open and read the file
                with open(file_path, 'r') as file:
                    author_identification = file.read()
            
            # load all the writting of that authors
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.1, random_state=42)
            
            # load sample text for mimicking
            # sample_text = ''
            # for idx, text in author_dataset.head().iterrows():
            #     sample_text += text['Question']+' '+ text['Answer']+ '\n\n'
            
            # read and select first 5 obfuscation samples as groundtruth
            mimicking_text = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round4_step2/'+filename.split('.')[0]+'.csv')
            sample_text = ''
            for idx, text in mimicking_text.head().iterrows():
                sample_text += text['Obfuscation']+ '\n\n'

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Question'] 
                  
                writing_sample = generate_synthesize_dataset(avg=290, author_identification=author_identification, sample_text=sample_text, input_text=input_text)
                    
                writing_sample = writing_sample.replace('\n',' ')
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(root_save+ filename.split('.')[0] +'.csv', index=False)

for dataname in ["speech","quora"]:
    mimicking_text(dataset_name=dataname)
