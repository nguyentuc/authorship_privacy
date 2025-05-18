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
        # client = OpenAI(
        #     # This is the default and can be omitted
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        # )

        # response = client.chat.completions.create(
        #     model=api,
        #     messages=[
        #         {
        #         "role": "developer",
        #         "content": [
        #             {
        #             "type": "text",
        #             "text": prompt,
        #             }
        #         ]
        #         }
        #     ],
        #     response_format={"type": "text"},
        #     seed=42,
        #     temperature=1.0,
        #     reasoning_effort="medium"
        # )

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        completion = client.chat.completions.create(
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
        return completion.choices[0].message.content

    #     return completion.choices[0].message.content

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

def verification(api, dataset_name, k_original):
    acc ={}

    if dataset_name=='speech':
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")
            
            # Select first 10 obfuscations texts of the author and 10 texts from others author: read from csv file
            df = pd.read_csv(synthesize_dataset+'obfuscation/'+person+'.csv')
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            sample_obfuscation_text = ''
            for text in df_shuffled['Obfuscation'][:10-k]:
                sample_obfuscation_text += text+ '\n\n'

            # select k from original
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            sample5 = author_dataset.select(range(k))
            for text in sample5:
                sample_obfuscation_text += text['text']+ '\n\n'
                
            # ramdonly select 50% for prediction
            remaining_indices = [i for i in range(len(author_dataset)) if i not in range(k)]
            filtered_dataset = author_dataset.select(remaining_indices)
            author_dataset = filtered_dataset.select(range(int(len(filtered_dataset) * 0.4)))
            

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
                attribution_result = authorship_verification(text_from_author=sample_obfuscation_text, text_from_other=sample_text_from_other, input_text=ip_text['text'], api= api)

                print(f"Text: {ip_text['text']}")
                attribution_result= attribution_result.strip().lower()
                print(f"Authorship verification: {i}/{len(author_dataset)}: {attribution_result}")

                if attribution_result=='yes':
                    count+=1
                records.append([ip_text['text'], attribution_result])
                print(80* '+')

            print(f"Accuracy: {count/len(author_dataset)}")
            acc[person] = count/len(author_dataset)
            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Input','Result'])
            # create folder to save
            save_path =  f'/media/volume/tucnv/Coding/AA/2b_evaluate_obfuscation_verification/{dataset_name}/{api}/without_user_metadata/attribution_from_{k_original}original_{10-k}obfuscation_10neg/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df_record.to_csv(save_path +person +'.csv', index=False)
        
        # save json file
        with open(f"/media/volume/tucnv/Coding/AA/2b_evaluate_obfuscation_verification/{dataset_name}/{api}/without_user_metadata/{k_original}_original.json", "w") as json_file:
            json.dump(acc, json_file, indent=4)

    elif dataset_name=='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'
        for filename in os.listdir(root_path):
            # Select randomly k original texts of the author
            author_dataset = pd.read_csv(f'/media/volume/tucnv/Coding/AA/Benchmark_generation/{dataset_name}/writing/'+filename.split('.')[0]+'.csv')
            sample_text = author_dataset.sample(n=k_original, random_state=42)
            sample_original_text = ''
            for idx, text in sample_text.iterrows():
                sample_original_text += text['Question']+' '+ text['Answer'].replace('\n','')+ '\n\n'
            
             # select obfuscation text for doing attribution
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset_name}/{api}/without_user_metadata/obfuscation/'+filename.split('.')[0]+'.csv')
            df_shuffled = df.sample(n=10-k_original, random_state=42).reset_index(drop=True)
            for idx, text in df_shuffled.iterrows():
                sample_original_text += text['Obfuscation'].replace('\n','')+ '\n\n'

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
            author_dataset = author_dataset.sample(frac=0.4, random_state=42)

            # doing authorship verification
            records = []
            i =0
            count = 0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                input_text = ip_text['Question']+' '+ ip_text['Answer'].replace('\n','')
                # verification process by LLMs
                # try:
                attribution_result = authorship_verification(text_from_author=sample_original_text, text_from_other=text_from_other, input_text=input_text, api= api)

                attribution_result = attribution_result.strip().lower()
                # except:
                #     attribution_result = 'yes'

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
            # create folder to save
            save_path =  f'/media/volume/tucnv/Coding/AA/2b_evaluate_obfuscation_verification/{dataset_name}/{api}/without_user_metadata/attribution_from_{k_original}original_{10-k}obfuscation_10neg/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df_record.to_csv(save_path +filename.split('.')[0] +'.csv', index=False)
        
        print(acc)
        # save json file
        with open(f"/media/volume/tucnv/Coding/AA/2b_evaluate_obfuscation_verification/{dataset_name}/{api}/without_user_metadata/{k_original}_original.json", "w") as json_file:
            json.dump(acc, json_file, indent=4)

# for api in ['deepseek', 'gemini']:
#     for name in ['quora', 'speech']:
#         for k in [0, 5, 10]:
#             verification(api=api, dataset_name=name, k_original= k)

for api in ['gemini']:
    for name in ['speech']:
        for k in [10]:
            verification(api=api, dataset_name=name, k_original= k)