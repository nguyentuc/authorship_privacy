import requests
import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Initialize an empty dictionary to hold the sums
from collections import defaultdict

def deepfakedetection(input_text):
    # Replace with your actual API key
    API_KEY = 'a74f7d46920c4d2a8864f853730a7b51'
    API_URL = 'https://api.gptzero.me/v2/predict/text'
    
    # Prepare the request payload
    payload = {
        "document": input_text,
        "version": "2025-05-02-base" 
    }
    # Set up the headers with your API key
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }

    # Send the POST request to GPTZero API
    response = requests.post(API_URL, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return 
        
    

def detecting(dataset_name):
    root_save =f"/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/deepfakedetection/round2_step2/"
    if dataset_name=='speech':
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")

            # original text
            # author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            # author_dataset = author_dataset.shuffle(seed=2025)
            # author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.1)))
            author_dataset = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/{dataset_name}/with_user_metadata/obfuscation/round2_step2/'+person+'.csv')

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ip_text['Obfuscation']  
                deepfakedetection_result = deepfakedetection(input_text)
            
                class_probability = deepfakedetection_result['documents'][0]['class_probabilities']
                print(f"Text: {input_text}")
                print(f"Detecting {i}/{len(author_dataset)}")
                print(class_probability)

                records.append([input_text, json.dumps(deepfakedetection_result), json.dumps(class_probability)])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Text', 'Detection_Logs', 'Class_Probs'])
            df_record.to_csv(root_save +person +'.csv', index=False)
    
    elif dataset_name =='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        for filename in os.listdir(root_path):
            
            # load all the writting of that authors
            # author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            # author_dataset = author_dataset.sample(frac=0.1, random_state=42)
            
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Loop_evaluation/quora/with_user_metadata/obfuscation/round2_step2/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.5, random_state=2025)
            

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Obfuscation']
                deepfakedetection_result = deepfakedetection(input_text)
                class_probability = deepfakedetection_result['documents'][0]['class_probabilities']
                
                print(f"Text: {input_text}")
                print(f"Detecting {i}/{len(author_dataset)}")
                print(class_probability)

                records.append([input_text, json.dumps(deepfakedetection_result), json.dumps(class_probability)])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Text', 'Detection_Logs', 'Class_Probs'])
            df_record.to_csv(root_save +filename.split('.')[0] +'.csv', index=False)

# for dataname in ["quora"]:
#     detecting(dataset_name=dataname)

def compute_average_score(file_path, folder):
    all_authors = []
    for filename in os.listdir(file_path):
        record = pd.read_csv(file_path+filename)
        all_class_probabilities = []
        for idx, detecting_prob in record.iterrows():
            class_probabilities = detecting_prob['Class_Probs']
            all_class_probabilities.append(json.loads(class_probabilities))
        
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)

        # Sum the values and count occurrences
        for d in all_class_probabilities:
            for key, value in d.items():
                sum_dict[key] += value
                count_dict[key] += 1

        # Compute the average
        average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
        all_authors.append(average_dict)
    # print(all_authors)
    
    # compute across all authors
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    # Sum the values and count occurrences
    for d in all_authors:
        for key, value in d.items():
            sum_dict[key] += value
            count_dict[key] += 1

    # Compute the average
    average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    print(f"{folder}: {average_dict}")
            

for folder in ['original', 'round1_step1', 'round1_step2', 'round2_step1', 'round2_step2']:
    compute_average_score(f'/media/volume/tucnv/Coding/AA/Loop_evaluation/speech/with_user_metadata/deepfakedetection/{folder}/', folder)


