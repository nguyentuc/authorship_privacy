import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Set up the API key
os.environ['OPENAI_API_KEY'] = "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"


# Function to ask ChatGPT to synthesize a user profile
def authorship_verification(author_name, author_identification, text_from_author, text_from_other, input_text, api):
    if api=='o3-mini':
        # Construct the prompt with variables
        prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by {author_name}. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
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
    
    elif api=='deepseek':
            prompt = f"You are a judge designed to verify the attribution of a human-author written text. You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by {author_name}. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}"
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

    elif api=='gemini':
        prompt = f"You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by {author_name}. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}."
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

    else:
        # Construct the prompt with variables
        prompt = f"You are given sample texts including 10 writings from the author and 10 writings from others. Analyze the writing styles of the input text, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. Your task is to verify if the input text was written by {author_name}. As output, exclusively return yes or no without any accompanying explanations or comments.\n\nHere is some information about the author: {author_identification}.\n\nThe 10 sample writings from the author: \n{text_from_author}\n\nThe 10 sample writing from other: \n{text_from_other}\n\nThe input text is: \n{input_text}."
        # print(prompt)
        # exit()
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
def verification(api, dataset):
    if dataset=='speech':
        synthesize_dataset = f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/'
        dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
        # print(f"Dataset structure: {dataset}")
        personalize_dataset = list(set(dataset['train']['style']))

        acc ={}
        for person in personalize_dataset:
            print(f"Working on:{person}")
            
            # Select first 2 mimicking texts of the author and 10 texts from others author: read from csv file
            df = pd.read_csv(synthesize_dataset+'mimicking_from_original/'+person+'.csv')
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            sample_mimicking_text = ''
            for text in df_shuffled['Mimicking'][:0]:
                sample_mimicking_text += text+ '\n\n'

            # select 8 from original
            author_dataset = dataset.filter(lambda example: example["style"] == person and len(example["text"].split()) > 50)['train']
            author_dataset = author_dataset.shuffle(seed=2024)
            author_dataset = author_dataset.shuffle(seed=2025)
            sample5 = author_dataset.select(range(10))
            
            for text in sample5:
                sample_mimicking_text += text['text']+ '\n\n'


            # sample text from other
            other_dataset = dataset.filter(lambda example: example["style"] != person and len(example["text"].split()) > 50)['train']
            other_dataset = other_dataset.shuffle(seed=2025)
            sample5 = other_dataset.select(range(10))
            sample_original_text_from_other = ''
            for text in sample5:
                sample_original_text_from_other += text['text']+ '\n\n'

            # author_identification 
            if person == 'trump':
                author_identification = "Donald Trump is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            elif person == 'obama':
                author_identification = "Barack Obama is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment"
            else:
                author_identification = "George W. Bush is a male speaker. His academic background is in Social Sciences. He is a native English speaker. He is from the United States, a native English-speaking (NS) environment" 

            # ramdonly select 50% for prediction
            remaining_indices = [i for i in range(len(author_dataset)) if i not in range(10)]
            filtered_dataset = author_dataset.select(remaining_indices)
            author_dataset = filtered_dataset.select(range(int(len(filtered_dataset) * 0.2)))
            records = []
            i =0
            count = 0
            for ip_text in author_dataset:
                i+= 1
                # verification process by LLMs
                attribution_result = authorship_verification(author_name=person, author_identification=author_identification, text_from_author=sample_mimicking_text, text_from_other=sample_original_text_from_other, input_text=ip_text['text'], api= api)
                

                print(f"Text: {ip_text['text']}")
                print(f"Authorship verification: {i}/{len(author_dataset)}")
                print(attribution_result)

                attribution_result = attribution_result.strip().lower()
                if attribution_result=='yes':
                    count+=1
                records.append([ip_text['text'], attribution_result])
                print(80* '+')

            print(f"Accuracy: {count/len(author_dataset)}")
            acc[person] = count/len(author_dataset)
            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Input','Result'])
            df_record.to_csv(f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/speech/{api}/with_user_metadata/attribution_from_original_10pos_10neg/' +person +'.csv', index=False)
        print(acc)

    elif dataset=='quora':
        # load all authors information
        root_path = '/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/user_profile/'

        acc ={}
        for filename in os.listdir(root_path):
            # User profile
            if filename.endswith('.txt'):  # Check if the file is a .txt file
                file_path = os.path.join(root_path, filename)
                
                # Open and read the file
                with open(file_path, 'r') as file:
                    author_identification = file.read()

            # Select randomly 10 original texts of the author
            author_dataset = pd.read_csv('/media/volume/tucnv/Coding/AA/Benchmark_generation/quora/writing/'+filename.split('.')[0]+'.csv')
            sample_text = author_dataset.sample(n=0, random_state=42)
            sample_original_text = ''
            for idx, text in sample_text.iterrows():
                sample_original_text += text['Question']+' '+ text['Answer'].replace('\n','')+ '\n\n'
            
             # select mimicking text for doing attribution
            df = pd.read_csv(f'/media/volume/tucnv/Coding/AA/1_evaluate_obfuscation_mimicking/{dataset}/{api}/with_user_metadata/mimicking_from_original/'+filename.split('.')[0]+'.csv')
            df_shuffled = df.sample(n=0, random_state=42).reset_index(drop=True)
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
                attribution_result = authorship_verification(author_name='the author', author_identification = author_identification, text_from_author=sample_original_text,  text_from_other=text_from_other, input_text=input_text, api=api)

                print(f"Text: {input_text}")
                print(f"Authorship verification: {i}/{len(author_dataset)}: {attribution_result.lower()}")

                attribution_result = attribution_result.strip().lower()
                if attribution_result=='yes':
                    count+=1
                records.append([input_text, attribution_result.lower()])
                print(80* '+')

            print(f"Accuracy: {count/len(author_dataset)}")
            acc[filename.split('.')[0]] = count/len(author_dataset)
            # saving synthesize dataset
            print("Saving")
            df_record = pd.DataFrame(records, columns=['Input','Result'])
            df_record.to_csv(f'/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/quora/deepseek/with_user_metada/attribution_from_0original_10mimicking_10neg/' +filename.split('.')[0] +'.csv', index=False)
        print(acc)

verification(api='deepseek',dataset='quora')