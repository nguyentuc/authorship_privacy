import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Set up the API key
# os.environ['OPENAI_API_KEY'] = "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"


# Function to ask ChatGPT to synthesize a user profile
def generate_synthesize_dataset(avg, sample_text, input_text):
    # Construct the prompt with variables
    prompt = f"You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from an author:\n{sample_text}\n\nThe input text is:\n{input_text}"
    # print(prompt)
    # exit()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )
    completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "text"},
            seed=42,
            temperature=1.0
    )
    return completion.choices[0].message.content

# from the obfuscation dataset of the author, randomly sampling 5 for sample writing.
# randomly 10% from the training set for mimicking process. 
# for each sample, call API for mimicking.
def mimicking_text(api, dataset):
    root_save =f"/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/additional_experiment/{dataset}/{api}/without_user_metadata/"
    if dataset=='speech':
        synthesize_dataset = f"/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/"
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")

            # read csv and select first 5 obfuscations as a condition text for mimicking
            df = pd.read_csv(synthesize_dataset+'/obfuscation/'+person+'.csv')
            sample_text = ''
            for text in df['Obfuscation'].head():
                sample_text += text+ '\n\n'

            # ramdonly select 20% for mimicking with 5 obfuscation text
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)

            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))

            records = []
            i =0
            for ip_text in author_dataset:
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ' '.join(ip_text['text'].split(' ')[:15])   

                writing_sample = generate_synthesize_dataset(avg=60, sample_text=sample_text, input_text=input_text)
                
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(root_save +person +'.csv', index=False)
    
    elif dataset =='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        for filename in os.listdir(root_path):
            # if filename.split(".")[0]+'.csv' in os.listdir('/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/additional_experiment/quora/4o-mini/with_user_metadata'):
            #     print(f"Skip:", filename.split(".")[0])
            #     continue

            # read csv and select first 5 obfuscations as a condition text for mimicking
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/obfuscation/'+filename.split('.')[0]+'.csv')
            sample_text = ''
            for text in df['Obfuscation'].head():
                sample_text += text.replace("\n", "")+ '\n\n'
            
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.4, random_state=42)

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Question']   
                try:
                    writing_sample = generate_synthesize_dataset(avg=290, sample_text=sample_text, input_text=input_text)
                except:
                    writing_sample = ip_text['Question']+' '+ ip_text['Answer']
                    
                writing_sample = writing_sample.replace('\n',' ')
                print(f"Original text: {input_text}")
                print(f"Obfuscation text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(root_save+ filename.split('.')[0] +'.csv', index=False)

for api in ["4o-mini","o3-mini", "deepseek", "gemini"]:
    for dataname in ["speech", "quora"]:
        mimicking_text(api=api, dataset=dataname)
