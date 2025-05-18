import openai
import os
import json
import pandas as pd
from openai import OpenAI
# Set up the API key
os.environ['OPENAI_API_KEY'] = "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"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function to ask ChatGPT to synthesize a user profile
# sample writings: Question:xxx, Answer:xxx
# sample question: Question:xxx
def generate_synthesize_dataset(user_profile, sample_writing, question):
    if user_profile == '':
        response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={ "type": "text" },
        seed=42,
        temperature=0.2,
        max_tokens=300,
        logprobs=True,
        messages=[
        {"role": "system", "content": "Imagine you are a Quora writer. Some writing samples were given, including questions and answers from that writer as follows: "+sample_writing+". Based on the given answer for each question, you first need to analyze some of the stylometric features of this writer such as lexical features (i.e: using the frequency of different character sequences, etc); syntactic features (i.e: part-of-speech distributions, occurrences of functional words and punctuation, etc); content features (i.e: semantics of words and phrases, etc).  You then need to analyze the Quora user writing based on style (i.e: direct, humorous, informal, etc); tone (i.e: casual, irreverent, speculative, etc); vocabulary (i.e: colloquial, pop-culture-infused, playful, etc); grammar (i.e: conversational, fluid, informal, etc); rhetorical devices (i.e: exaggeration, humor, casual asides) ; content patterns  (i.e: character analysis, hypothetical scenarios, contrasts); values (i.e: pragmatism, realism, creativity); etc."},
        {"role": "system", "content":"Based on your analysis, you need to mimic the writing style of this Quora writer to provide an answer to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

    elif sample_writing == '':
        response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={ "type": "text" },
        seed=42,
        temperature = 0.2,
        max_tokens=300,
        logprobs=True,
        messages=[
        # all information
        # {"role": "system", "content": "Imagine you are a Quora writer "+ user_profile['personal_info']['name'] +" with some background information described below: Location:"+ str(user_profile['personal_info']['location']) if user_profile['personal_info']['location'] != None else '' +" Age: " + str(user_profile['personal_info']['age']) if user_profile['personal_info']['age'] !=None else '' + " Gender: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +" Education:" +str(user_profile['education']) if user_profile['education'] != None else ''+" Professional Experience: "+str(user_profile['professional_experience']) if user_profile['professional_experience'] != None else ''+" Skill: "+str(user_profile['skills']) if user_profile['skills'] != None else ''+". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # remove name feature in prompt
        # {"role": "system", "content": "Imagine you are a Quora writer with some background information described below: Location:"+ str(user_profile['personal_info']['location']) if user_profile['personal_info']['location'] != None else '' +" Age: " + str(user_profile['personal_info']['age']) if user_profile['personal_info']['age'] !=None else '' + " Gender: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +" Education:" +str(user_profile['education']) if user_profile['education'] != None else ''+" Professional Experience: "+str(user_profile['professional_experience']) if user_profile['professional_experience'] != None else ''+" Skill: "+str(user_profile['skills']) if user_profile['skills'] != None else ''+". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # remove location feature
        # {"role": "system", "content": "Imagine you are a Quora writer "+ user_profile['personal_info']['name'] +" with some background information described below: Age: " + str(user_profile['personal_info']['age']) if user_profile['personal_info']['age'] !=None else '' + " Gender: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +" Education:" +str(user_profile['education']) if user_profile['education'] != None else ''+" Professional Experience: "+str(user_profile['professional_experience']) if user_profile['professional_experience'] != None else ''+" Skill: "+str(user_profile['skills']) if user_profile['skills'] != None else ''+". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # remove age
        # {"role": "system", "content": "Imagine you are a Quora writer "+ user_profile['personal_info']['name'] +" with some background information described below: Location:"+ str(user_profile['personal_info']['location']) if user_profile['personal_info']['location'] != None else '' + " Gender: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +" Education:" +str(user_profile['education']) if user_profile['education'] != None else ''+" Professional Experience: "+str(user_profile['professional_experience']) if user_profile['professional_experience'] != None else ''+" Skill: "+str(user_profile['skills']) if user_profile['skills'] != None else ''+". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # name only
        # {"role": "system", "content": "Imagine you are a Quora writer "+ user_profile['personal_info']['name'] +". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # location only
        # {"role": "system", "content": "Imagine you are a Quora writer and your current location is: "+ str(user_profile['personal_info']['location']) if user_profile['personal_info']['location'] != None else "" +". You first need to analyze the user background."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        # age only
        # {"role": "system", "content": "Imagine you are a Quora writer with age: " + str(user_profile['personal_info']['age']) if user_profile['personal_info']['age'] !=None else '' +". You first need to analyze the user information."},
        # {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])

        {"role": "system", "content": "Imagine you are a Quora writer with gender is: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +". You first need to analyze the user information."},
        {"role":"system", "content": "You need to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])



    else:
        response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={ "type": "text" },
        seed=42,
        temperature =0.2,
        max_tokens=300,
        logprobs=True,
        messages=[
        {"role": "system", "content": "Imagine you are a Quora writer "+ user_profile['personal_info']['name'] +", with some background information described below: Location:"+ str(user_profile['personal_info']['location']) if user_profile['personal_info']['location'] != None else '' +" Age: " + str(user_profile['personal_info']['age']) if user_profile['personal_info']['age'] !=None else '' + " Gender: " + user_profile['personal_info']['gender'] if user_profile['personal_info']['gender'] != None else '' +" Education:" +str(user_profile['education']) if user_profile['education'] != None else '' +" Professional Experience: "+str(user_profile['professional_experience']) if user_profile['professional_experience'] != None else ''+" Skill: "+str(user_profile['skills']) if user_profile['skills'] != None else '' +". Some writing samples were given, including questions and answers from "+ user_profile['personal_info']['name'] +" as follows: "+sample_writing+". You first need to analyze user background. Then, based on the given answer for each question, you need to analyze some of the stylometric features of this writer such as lexical features (i.e: using the frequency of different character sequences, etc); syntactic features (i.e: part-of-speech distributions, occurrences of functional words and punctuation, etc); content features (i.e: semantics of words and phrases, etc).  Next, you need to analyze the Quora user writing based on style (i.e: direct, humorous, informal, etc); tone (i.e: casual, irreverent, speculative, etc); vocabulary (i.e: colloquial, pop-culture-infused, playful, etc); grammar (i.e: conversational, fluid, informal, etc); rhetorical devices (i.e: exaggeration, humor, casual asides) ; content patterns  (i.e: character analysis, hypothetical scenarios, contrasts); values (i.e: pragmatism, realism, creativity); etc."},
        {"role":"system", "content":"You need to mimic the writing style of this Quora writer to provide an answer between 250 and 300 words to this question: "+question+". Please only output the answer in a paragraph and nothing else."}])
    return response.choices[0].logprobs.content, response.choices[0].message.content

# get the list of 200 authors who have both writing and profile for training
profile_path ='/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/converted_user_profile/'
quora_writing = '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/train_40_5_5/'
synthesize_dataset = '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/synthesize_dataset_v4_prompt_gender_only_temp_0.2/'
for file_name in os.listdir(profile_path):
    # check to make sure we have generate wiritng for that author then skip
    if file_name.split('.')[0]+'.csv' in os.listdir(synthesize_dataset):
        records_file = pd.read_csv(synthesize_dataset+ file_name.split('.')[0]+'.csv')
        if records_file.shape[0] >= 10:
            print("Skip userid: ", file_name.split('.')[0])
            continue
        
    file_path = os.path.join(profile_path, file_name)
    # get sample writing
    with open(file_path, 'r') as f:
        user_profile = json.load(f)
        user_writing = quora_writing + file_name.split('.')[0]+'.csv'
        user_writing = pd.read_csv(user_writing)
        sample_rows = user_writing.sample(n=5, random_state=2024)
        sample_writing = ''
        for idx, row in sample_rows.iterrows():
            # make sample writing:
            sample_writing += 'Question: '
            sample_writing += str(row['Question'])
            sample_writing += '\n'
            sample_writing += 'Answer: '
            sample_writing += str(row['Answer'].replace('\t', ' '))
            sample_writing += '\n'
        
        records = []
        # generate synthesize dataset on the first 10 samples
        for idx, row in user_writing[:10].iterrows():
            question = str(row['Question'])
            print("Synthesizing: ",question)

            # generate with sample writing only
            # log_prob1, writing_sample1 = generate_synthesize_dataset(user_profile='', sample_writing=sample_writing, question=question)
            # save_log_prob1 = []
            # for item in log_prob1:
            #     save_log_prob1.append(str(item.token)+"+;+"+ str(item.logprob))
            # save_log_prob1 = "|;|".join(save_log_prob1)
            # print("Sample 1: ", writing_sample1)
            

            # generate with user profile only
            log_prob2, writing_sample2 = generate_synthesize_dataset(user_profile= user_profile, sample_writing='', question=question)
            save_log_prob2 = []
            for item in log_prob2:
                save_log_prob2.append(str(item.token)+"+;+"+ str(item.logprob))
            save_log_prob2 = "|;|".join(save_log_prob2)
            print("Synthesize user profile: ", writing_sample2)

            # generate synthesize dataset full information
            # log_prob3, writing_sample3 = generate_synthesize_dataset(user_profile=user_profile, sample_writing=sample_writing, question=question)
            # save_log_prob3 = []
            # for item in log_prob3:
            #     save_log_prob3.append(str(item.token)+"+;+"+ str(item.logprob))
            # save_log_prob3 = "|;|".join(save_log_prob3)
            # print("Sample 3: ", writing_sample3)

            records.append([row['Name'], question, row['Answer'], '', '', writing_sample2, save_log_prob2, '', '', None])
            print(80* '+')

        # saving synthesize dataset
        print("Saving")
        df_record = pd.DataFrame(records, columns=['Name', 'Question', 'Old_Answer', 'Answer_with_writing_sample','Log_prob_writing_sample', 'Answer_with_user_profile','Log_prob_user_profile', 'Answer_with_full_information','Log_prob_full_information', 'Image'])
        df_record.to_csv(synthesize_dataset+ file_name.split('.')[0]+'.csv', index=False)
