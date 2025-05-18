import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Set up the API key
# os.environ['OPENAI_API_KEY'] = "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"

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# Function to ask ChatGPT to synthesize a user profile
def generate_text(avg, sample_text, input_text, api):
    if api=='gemini':
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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

    elif api=='o3-mini':
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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

    elif api=='deepseek':
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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

# from the original dataset of the author, randomly sampling 5 for sample writing
# randomly 20% from the training set for obfuscation process 
# for each sample, call API for obfuscation
def text_mimicking(api, dataset_name):
    if dataset_name=='speech':
        path = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/{dataset_name}/{api}/without_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/without_user_metadata/mimicking_from_incorrect_attribute/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        personalize_dataset = list(set(dataset['train']['style']))

        for person in personalize_dataset:
            print(f"Working on:{person}")

            # choose 5 sample writing that are correctly classified
            df = pd.read_csv(path+person+'.csv')
            df_filter = df[df['Result']=='no'].sample(n=5, random_state=2025)
        
            sample_correct_attribution_text = ''
            for idx, row in df_filter.iterrows():
                sample_correct_attribution_text += row['Input']+ '\n\n'
            
            # Get 20% from training set for obfuscation
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

                # generate with user profile only
                writing_sample = generate_text(avg=60, sample_text=sample_correct_attribution_text, input_text=input_text, api=api)

                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(save_path+ person +'.csv', index=False)

    elif dataset_name=='quora':
        path_correct_attribute = f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/{dataset_name}/{api}/without_user_metadata/attribution_from_original_10pos_10neg/'
        save_path = f'/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/{dataset_name}/{api}/without_user_metadata/mimicking_from_incorrect_attribute/'
        # get list of authors
        root_path = f'/media/volume/tucnv/Coding/AA/Benchmark_generation/{dataset_name}/user_profile/'
        for filename in os.listdir(root_path):
            # if filename.split('.')[0]+'.csv' in os.listdir('/media/volume/tucnv/Coding/AA/3b_evaluate_verification_mimicking/quora/deepseek/without_user_metadata/mimicking_from_incorrect_attribute'):
            #     print('Skip:', filename.split('.')[0]+'.csv')
            #     continue
            # choose 5 sample writing that are correctly classified
            df = pd.read_csv(path_correct_attribute+filename.split('.')[0]+'.csv')
            df_filter = df[df['Result']=='no'].sample(n=5, random_state=42, replace=True)
        
            sample_attribution_text = ''
            for idx, row in df_filter.iterrows():
                sample_attribution_text += row['Input'].replace('\n',' ')+ '\n\n'
            

            # Get 20% from original for mimicking
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            author_dataset = author_dataset.sample(frac=0.2, random_state=42)
            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ip_text['Question']

                # generate with user profile only
                writing_sample = generate_text(avg=290, sample_text=sample_attribution_text, input_text=input_text, api=api)

                writing_sample=writing_sample.replace('\n', '')
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(save_path+ filename.split('.')[0] +'.csv', index=False)

for api in ['gemini','o3-mini']:
    for dataset in ['speech','quora']:
        text_mimicking(api=api, dataset_name= dataset)

# for api in ['deepseek']:
#     for dataset in ['quora']:
#         text_mimicking(api=api, dataset_name= dataset)
