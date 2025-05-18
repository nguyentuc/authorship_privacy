import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
import backoff 
import openai
from openai import OpenAI

# Set up the API key
os.environ['OPENAI_API_KEY'] = "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"



# Function to ask ChatGPT to synthesize a user profile
def authorship_verification(text_from_author, text_from_other, input_text, api):

    if api=='o3-mini':
        # Construct the prompt with variables
        prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by the author. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
        # print(prompt)
        # exit()
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        response = client.chat.completions.create(
            model=api,
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
        return response.choices[0].message.content

    elif api=='gemini':
        prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by the author. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
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

    elif api=='deepseek':
            prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by the author. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
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

    else:
        # Construct the prompt with variables
        prompt = f"You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by the author. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
        # print(prompt)
        # exit()
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={ "type": "text" },
        seed=42,
        temperature=1.0,
        max_tokens=300,
        logprobs=True,
        messages=[
            {
            "role": "system",
            "content": "You are a judge designed to verify the attribution of a human-author written text."
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        )
        return response.choices[0].message.content

# from the obfuscation dataset of the author, randomly sampling 5 for sample writing.
# randomly 10% from the training set for mimicking process. 
# for each sample, call API for mimicking.

def verification(api, dataset_name):
    if dataset_name=='speech':
        acc ={}
        for num_original in [0,2,5,8,10]:
            # get the obfuscation dataset
            synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
            dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
            personalize_dataset = list(set(dataset['train']['style']))

            acc[num_original] ={}
            for person in personalize_dataset:
                print(f"Working on:{person}")
                
                # Select first 10 obfuscation texts of the author and 10 texts from others author: read from csv file
                df = pd.read_csv(synthesize_dataset+'/obfuscation/'+person+'.csv')
                sample_pos_text = ''
                for text in df['Obfuscation'][:10-num_original]:
                    sample_pos_text += text+ '\n\n'

                # select 10 from original
                author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
                author_dataset = author_dataset.shuffle(seed=2024)
                author_dataset = author_dataset.shuffle(seed=2025)
                sample5 = author_dataset.select(range(num_original))
                for text in sample5:
                    sample_pos_text += text['text']+ '\n\n'
                    
                # ramdonly select 10% for prediction
                remaining_indices = [i for i in range(len(author_dataset)) if i not in range(num_original)]
                filtered_dataset = author_dataset.select(remaining_indices)
                author_dataset = filtered_dataset.select(range(int(len(filtered_dataset) * 0.1)))
                

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
                    attribution_result = authorship_verification(text_from_author=sample_pos_text, text_from_other=sample_text_from_other, input_text=ip_text['text'], api= api)

                    print(f"Text: {ip_text['text']}")
                    print(f"Authorship verification: {i}/{len(author_dataset)}")
                    print(attribution_result)

                    attribution_result= attribution_result.strip().lower()

                    if attribution_result=='yes':
                        count+=1
                    records.append([ip_text['text'], attribution_result])
                    print(80* '+')

                print(f"Accuracy: {count/len(author_dataset)}")
                acc[num_original][person] = count/len(author_dataset)
                # saving synthesize dataset
                print("Saving")
                df_record = pd.DataFrame(records, columns=['Input','Result'])
                save_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/speech/{api}/without_user_metadata/attribution_from_{num_original}original_{10-num_original}obfuscation/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                df_record.to_csv(save_path+person+'.csv', index=False)
        print(acc)

    elif dataset_name=='quora':
        acc ={}
        for num_original in [0,5,10]:
            # load all authors information
            root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
            synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
            acc[num_original] ={}
            for filename in os.listdir(root_path):

                # Select randomly 10 original texts of the author
                author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
                sample_text = author_dataset.sample(n=num_original, random_state=42)
                sample_pos_text = ''
                for idx, text in sample_text.iterrows():
                    sample_pos_text += text['Question']+' '+ text['Answer'].replace('\n','')+ '\n\n'

                
                # select obfuscation text for doing attribution
                df = pd.read_csv(synthesize_dataset+'/obfuscation/'+filename.split('.')[0]+'.csv')
                sample_pos_text = ''
                for text in df['Obfuscation'][:10-num_original]:
                    sample_pos_text += text.replace('\n','')+ '\n\n'

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

                # remaining text
                author_dataset = author_dataset.drop(sample_text.index)
                author_dataset = author_dataset.sample(frac=0.2, random_state=42)

                # doing authorship verification
                records = []
                i =0
                count = 0
                for idx, ip_text in author_dataset.iterrows():
                    i+= 1
                    input_text = ip_text['Question']+' '+ ip_text['Answer'].replace('\n','')
                    # verification process by LLMs
                    # try:
                    attribution_result = authorship_verification(text_from_author=sample_pos_text, text_from_other=text_from_other, input_text=input_text, api= api)
                    # except:
                    #     attribution_result ='no'

                    attribution_result = attribution_result.strip().lower()
                    print(f"Text: {input_text}")
                    print(f"Authorship verification: {i}/{len(author_dataset)}: {attribution_result}")
                    
                    if attribution_result=='yes':
                        count+=1
                    records.append([input_text, attribution_result])

                    print(80* '+')

                print(f"Accuracy: {count/len(author_dataset)}")
                acc[num_original][filename.split('.')[0]] = count/len(author_dataset)
                # saving synthesize dataset
                print("Saving")
                df_record = pd.DataFrame(records, columns=['Input','Result'])
                save_path = f'/media/volume/tucnv/Coding/AA/4_evaluate_obfuscation_attribution/quora/{api}/without_user_metadata/attribution_from_{num_original}original_{10-num_original}obfuscation/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                df_record.to_csv(save_path +filename.split('.')[0] +'.csv', index=False)
        print(acc)

verification(api='o3-mini', dataset_name='quora')