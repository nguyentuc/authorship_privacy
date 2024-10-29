import openai
import tiktoken  # Tokenizer compatible with GPT models (use tiktoken library)
import os 
from openai import OpenAI

# Set up your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load the tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")  # Assuming GPT-4, you can change this based on the model

def get_next_token_prediction(prompt):
    # Use the ChatGPT API to predict the next token based on the current sequence
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,  # No randomness
        logprobs=1  # Get top log probabilities for the tokens
    )
    return response

def calculate_approx_perplexity(text):
    # Tokenize the input text
    tokens = encoding.encode(text)
    
    # Initialize variables for perplexity calculation
    total_log_prob = 0.0
    token_count = len(tokens)
    
    # Loop through the tokens to predict the next token
    for i in range(1, token_count):
        prompt = encoding.decode(tokens[:i])  # Take the preceding tokens as context
        response = get_next_token_prediction(prompt)
        
        # Extract the log probability of the actual next token
        actual_next_token = tokens[i]
        logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        
        # Get the log probability of the actual next token
        actual_next_token_str = encoding.decode([actual_next_token])
        log_prob = logprobs.get(actual_next_token_str, float('-inf'))
        
        total_log_prob += log_prob
    
    # Calculate perplexity
    avg_log_prob = total_log_prob / token_count
    perplexity = 2 ** (-avg_log_prob)
    
    return perplexity

# Input text for which you want to calculate perplexity
text = "This is a sample text for testing perplexity."

# Calculate perplexity (approximation)
perplexity = calculate_approx_perplexity(text)
print(f'Approximate Perplexity: {perplexity}')
