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
def generate_synthesize_profile(json_template, quora_profile):
    response = client.chat.completions.create(
    model="gpt-4o-mini", 
    response_format={ "type": "json_object" },
    seed=42,
    messages=[
    {"role": "system", "content": "You are an expert text analyst. Given the user profile template: "+str(json_template)+". Please extract the most relevant keywords and key phrases from this Quora user information: "+str(quora_profile)+" to fill out the user profile template. If you can not extract any information for features in the profile template, just return "". Return the result as a JSON file and nothing else."}])
    return response.choices[0].message.content

# get the list of 200 authors who have both writing and profile for training
json_template ='/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/user_profile.json'
with open(json_template, 'r') as file:
    json_profile_template = json.load(file)


quora_user_profile = '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/user_profile/'

for file_name in os.listdir(quora_user_profile):
    user_profile = quora_user_profile+ file_name
    with open(user_profile, 'r') as file:
        user_profile = json.load(file)
    
    response = generate_synthesize_profile(json_profile_template, user_profile)
    with open('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/converted_user_profile/'+ file_name.split('.')[0]+'.json', 'w') as file:
        print("Writing: ", file_name)
        file.write(response)