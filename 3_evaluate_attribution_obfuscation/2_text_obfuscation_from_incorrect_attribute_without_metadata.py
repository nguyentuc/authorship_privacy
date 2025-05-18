import openai
import os
import json
import pandas as pd
from openai import OpenAI
from datasets import load_from_disk
# Set up the API key
os.environ['OPENAI_API_KEY'] = "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"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function to ask ChatGPT to synthesize a user profile
def generate_obfuscation_text(avg, sample_text, input_text):
    # Construct the prompt with variables
    prompt = f"You are given 5 sample writings from an author. The goal of this task is to conceal the author's writing style by carefully modifying lexical richness and diversity, sentence structure, punctuation patterns, special character usage, expressions and idioms, overall tone, emotion, mood, and any other distinguishing stylistic elements. Your task is to generate {avg}-word continuation that has writing style significantly different from the provided input text. Strive to make the rewritten text distinguishable from both the input text and the 5 sample writings by the author. As output, exclusively return the text completion without any accompanying explanations or comments.\n\nThe 5 sample writings from an author:\n{sample_text}\n\nThe input text is:\n{input_text}"
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
        "content": "You are an emulator designed to hide the writing style of a human author."
        },
        {
        "role": "user",
        "content": prompt
        }
    ],
    )
    return response.choices[0].logprobs.content, response.choices[0].message.content

# from the original dataset of the author, randomly sampling 5 for sample writing
# randomly 20% from the training set for obfuscation process 
# for each sample, call API for obfuscation

path = '/media/volume/tucnv/Coding/AA/2_evaluate_mimicking_attribution/without_user_metadata/attribution_from_original_10pos_10neg/'
save_path = '/media/volume/tucnv/Coding/AA/3_evaluate_attribution_obfuscation/without_user_metadata/obfuscation_from_incorrect_attribute/'
dataset = load_from_disk("/media/volume/tucnv/Coding/AA/Benchmark_generation/speech")
personalize_dataset = list(set(dataset['train']['style']))

for person in personalize_dataset:
    print(f"Working on:{person}")

    # choose 5 sample writing that are incorrectly classified
    df = pd.read_csv(path+person+'.csv')
    df_filter = df[df['Result']=='no'].sample(n=5, random_state=2025)
   
    sample_incorrect_attribution_text = ''
    for idx, row in df_filter.iterrows():
        sample_incorrect_attribution_text += row['Input']+ '\n\n'

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
        log_prob, writing_sample = generate_obfuscation_text(avg=60, sample_text=sample_incorrect_attribution_text, input_text=input_text)
        save_log_prob = []
        for item in log_prob:
            save_log_prob.append(str(item.token)+"+;+"+ str(item.logprob))
        save_log_prob = "|;|".join(save_log_prob)

        print(f"Original text: {input_text}")
        print(f"Obfuscation text {i}/{len(author_dataset)}")
        print(input_text+ ' ' +writing_sample)

        records.append([input_text+' '+writing_sample, save_log_prob])
        print(80* '+')

    # saving synthesize dataset
    print("Saving")
    df_record = pd.DataFrame(records, columns=['Obfuscation','Log_prob'])
    df_record.to_csv(save_path+ person +'.csv', index=False)
