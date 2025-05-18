import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Set up the API key
os.environ['OPENAI_API_KEY'] = "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"


# Function to ask ChatGPT to synthesize a user profile
def generate_synthesize_dataset(avg, author_identification, sample_text, input_text, api):
    if api=='o3-mini':
        # Construct the prompt with variables
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
        # print(prompt)
        # exit()
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),)
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

    elif api=='deepseek':
        prompt = f"You are an emulator designed to replicate the writing style of a human author. You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 5 sample writings from the author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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
        prompt = f"You are given 5 sample writings from the author. The goal of this task is to mimic the author’s writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion, and mood, or any other relevant aspect of writing style established by the author. Your task is to generate a {avg}-word continuation that seamlessly blends with the provided input text. Ensure that the continuation is indistinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 5 sample writings from an author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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
        max_tokens=500,
        logprobs=True,
        messages=[
            {
            "role": "system",
            "content": "You are an emulator designed to replicate the writing style of a human author."
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
def mimicking_text(api, dataset):
    root_save =f"/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/additional_experiment/{dataset}/{api}/with_user_metadata/"
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

            # author_identification 
            if person == 'trump':
                author_identification = "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            elif person == 'obama':
                author_identification = "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            else:
                author_identification = "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 

            # ramdonly select 20% for mimicking with 5 obfuscation text
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)

            # sample5 = author_dataset.select(range(5))
            # sample_text = ''
            # for text in sample5:
            #     sample_text += text['text']+ '\n\n'

            author_dataset = author_dataset.select(range(int(len(author_dataset) * 0.2)))

            records = []
            i =0
            for ip_text in author_dataset:
                i+= 1
                # get the first 15 words for continue obfuscaton
                input_text = ' '.join(ip_text['text'].split(' ')[:15])   

                writing_sample = generate_synthesize_dataset(avg=60, author_identification=author_identification, sample_text=sample_text, input_text=input_text, api=api)
                
                print(f"Original text: {input_text}")
                print(f"Mimicking text {i}/{len(author_dataset)}")
                print(input_text+ ' ' +writing_sample)

                records.append([input_text+' '+writing_sample])
                print(80* '+')

            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Mimicking'])
            df_record.to_csv(synthesize_dataset+'mimicking_from_original/' +person +'.csv', index=False)
    
    elif dataset =='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        for filename in os.listdir(root_path):
            if filename.endswith('.txt'):  # Check if the file is a .txt file
                file_path = os.path.join(root_path, filename)
                
                # Open and read the file
                with open(file_path, 'r') as file:
                    author_identification = file.read()

            # read csv and select first 5 obfuscations as a condition text for mimicking
            # df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/obfuscation/'+filename.split('.')[0]+'.csv')
            # sample_text = ''
            # for text in df['Obfuscation'].head():
            #     sample_text += text.replace("\n", "")+ '\n\n'
            

            # read the original question and ask for obfuscation to answer the question
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            sample_text = ''
            for idx, text in author_dataset.head().iterrows():
                sample_text += text['Question']+' '+ text['Answer']+ '\n\n'
            
            author_dataset = author_dataset.sample(frac=0.4, random_state=42)

            records = []
            i =0
            for idx, ip_text in author_dataset.iterrows():
                i+= 1
                # get the question
                input_text = ip_text['Question']   
                try:
                    writing_sample = generate_synthesize_dataset(avg=290, author_identification=author_identification, sample_text=sample_text, input_text=input_text, api=api)
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
            df_record = pd.DataFrame(records, columns=['Obfuscation'])
            df_record.to_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/mimicking_from_original/'+ filename.split('.')[0] +'.csv', index=False)

for api in ["4o-mini", "o3-mini", "gemini", "deepseek"]:
    for dataname in ["speech","quora"]:
        mimicking_text(api=api, dataset=dataname)
