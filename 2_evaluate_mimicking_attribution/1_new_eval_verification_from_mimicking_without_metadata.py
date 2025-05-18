import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
import backoff 
import openai
from openai import OpenAI

# Function to ask ChatGPT to synthesize a user profile
def authorship_verification(text_from_author, text_from_other, input_text):
    prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by the author. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(
        model="openai/o3-mini",
        messages=[
            {
            "role": "developer",
            "content": [
                {
                "type": "text",
                "text": prompt,
                }
            ]
            }
        ],
        response_format={"type": "text"},
        seed=42,
        temperature=1.0,
        reasoning_effort="medium"
    )
    print(completion)
    return completion.choices[0].message.content

# from the obfuscation dataset of the author, randomly sampling 5 for sample writing.
# randomly 10% from the training set for mimicking process. 
# for each sample, call API for mimicking.

def verification(api, dataset_name):
    save_folder = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/additional_experiment/{dataset_name}/{api}/without/'
    if dataset_name=='speech':
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/with_user_metadata/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        acc ={}
        for person in personalize_dataset:
            print(f"Working on:{person}")
            
            # Select first 10 mimicking texts of the author and 10 texts from others author: read from csv file
            df = pd.read_csv(synthesize_dataset+'mimicking_from_original/'+person+'.csv')
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            sample_mimicking_text = ''
            for text in df_shuffled['Mimicking'][:10]:
                sample_mimicking_text += text+ '\n\n'

            # ramdonly select 20% for prediction
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))
                
            # sample text from other
            other_dataset = dataset.filter(lambda example: example["style"] != person and len(example["text"].split()) > 50)['train']
            other_dataset = other_dataset.shuffle(seed=2025)
            sample10 = other_dataset.select(range(10))
            sample_text_from_other = ''
            for text in sample10:
                sample_text_from_other += text['text']+ '\n\n'


            records = []
            i =0
            count = 0
            for ip_text in author_dataset:
                i+= 1
                # verification process by LLMs
                try:
                    attribution_result = authorship_verification(text_from_author=sample_mimicking_text, text_from_other=sample_text_from_other, input_text=ip_text['text'])
                    attribution_result= attribution_result.strip().lower()
                except:
                    attribution_result = 'no'

                print(f"Text: {ip_text['text']}")
                print(f"Authorship verification: {i}/{len(author_dataset)}")
                print(attribution_result)
                    
                if attribution_result=='yes':
                    count+=1
                records.append([ip_text['text'], attribution_result])
                print(80* '+')

            print(f"Accuracy: {count/len(author_dataset)}")
            acc[person] = count/len(author_dataset)
            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Input','Result'])
            df_record.to_csv(save_folder +person +'.csv', index=False)
        print(acc)
        with open(f"/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/additional_experiment/{dataset_name}/{api}/without/results.json", "w") as json_file:
            json.dump(acc, json_file, indent=4)

    elif dataset_name=='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        acc ={}
        for filename in os.listdir(root_path):
            
            # select mimicking text for doing attribution
            sample_original_text = ''
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/quora/{api}/with_user_metadata/mimicking_from_original/'+filename.split('.')[0]+'.csv')
            df_shuffled = df.sample(n=10, random_state=42).reset_index(drop=True)
            for idx, text in df_shuffled.iterrows():
                sample_original_text += text['Mimicking'].replace('\n','')+ '\n\n'

            # sample 10 from others for attribution
            other_authors = list(set(os.listdir(root_path)) - set(filename))
            all_writing = []
            for other_author in other_authors:
                writing= pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+other_author.split('.')[0]+'.csv')
                all_writing.append(writing)
            merged_other_writings = pd.concat(all_writing, ignore_index=True)
            merged_other_writings = merged_other_writings.sample(frac=1, random_state=42).reset_index(drop=True)
            negative_sampling = merged_other_writings.sample(n=10, random_state=42)
            text_from_other = ''
            for idx, row in negative_sampling.iterrows():
                text_from_other += row['Question']+' '+ row['Answer'].replace('\n','')+ '\n\n'

            # sample text for verification
            author_dataset = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Benchmark_generation/{dataset_name}/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)

            # doing authorship verification
            records = []
            i =0
            count = 0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                input_text = ip_text['Question']+' '+ ip_text['Answer'].replace('\n','')
                # verification process by LLMs
                try:
                    attribution_result = authorship_verification(text_from_author=sample_original_text, text_from_other=text_from_other, input_text=input_text)
                    attribution_result = attribution_result.strip().lower()
                except:
                    attribution_result= 'no'
                    
                print(f"Text: {input_text}")
                print(f"Authorship verification: {i}/{len(author_dataset)}: {attribution_result}")
                
                if attribution_result=='yes':
                    count+=1
                records.append([input_text, attribution_result])

                print(80* '+')

            print(f"Accuracy: {count/len(author_dataset)}")
            acc[filename.split('.')[0]] = count/len(author_dataset)
            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Input','Result'])
            df_record.to_csv(save_folder +filename.split('.')[0] +'.csv', index=False)
        print(acc)
        with open(f"/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/additional_experiment/{dataset_name}/{api}/without/results.json", "w") as json_file:
            json.dump(acc, json_file, indent=4)

for api in ['gemini']:
    for dataset_name in ['speech']:
        verification(api=api, dataset_name=dataset_name)