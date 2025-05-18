import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk

# Function to ask ChatGPT to synthesize a user profile
def generate_synthesize_dataset(avg, sample_text, input_text):
    # Construct the prompt with variables
    prompt = f"You are an emulator designed to hide the writing style of a human author. You are given 5 sample writings from an author. The goal of this task is to conceal the author's writing style by carefully modifying lexical richness and diversity, sentence structure, punctuation patterns, special character usage, expressions and idioms, overall tone, emotion, mood, and any other distinguishing stylistic elements. Your task is to generate {avg}-word continuation that has writing style significantly different from the provided input text. Strive to make the rewritten text distinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
    # print(prompt)
    # exit()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )
    response = client.chat.completions.create(
        model="openai/o3-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={"type": "text"},
        seed=42,
        temperature=1.0,
        reasoning_effort="medium"
    )
    return response.choices[0].message.content


# from the original dataset of the author, randomly sampling 5 for sample writing
# randomly 20% from the training set for obfuscation process 
def obfuscation_text(api, dataset_name):
    save_path = f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/additional_experiment/{dataset_name}/{api}/without/'

    if dataset_name == 'speech':
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            if person+'.csv' in os.listdir(save_path):
                print("Skip: ", person)
                continue
            print(f"Working on:{person}")

            # read 5 mimicking examples
            mimicking_sample = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/with_user_metadata/micking_sample/'+person+'.csv')
            sample_text = ''
            for _, text in mimicking_sample.iterrows():
                sample_text += text['Mimicking']+ '\n\n'

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
                try:
                    writing_sample = generate_synthesize_dataset(avg=60, sample_text=sample_text, input_text=input_text)
                except:
                     writing_sample = ip_text['text']  
                     
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
            # if filename.split('.')[0]+'.csv' in os.listdir(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/additional_experiment/quora/4o-mini/without/'):
            #     print("Skipping:", filename.split('.')[0])
            #     continue


            # load 5 mimicking example for fewshort obfuscation
            mimicking_sample = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1b_evaluate_mimicking_influence_obfuscation/{dataset_name}/{api}/with_user_metadata/micking_sample/'+filename.split('.')[0]+'.csv')
            sample_text = ''
            for _, text in mimicking_sample.iterrows():
                sample_text += text['Mimicking']+ '\n\n'
            
            # load all the writting of that authors
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Question']   
                try:
                    writing_sample = generate_synthesize_dataset(avg=290, sample_text=sample_text, input_text=input_text)
                    
                    writing_sample =writing_sample.replace('\n', ' ')
                except:
                    writing_sample = ip_text['Answer']
                print(f"Original text: {input_text}")
                print(f"Obfuscation text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Obfuscation'])
            df_record.to_csv(save_path+ filename.split('.')[0] +'.csv', index=False)

for api in ["o3-mini", "gemini", "deepseek"]:
    for dataset_name in ['quora']:
        obfuscation_text(api=api, dataset_name=dataset_name)