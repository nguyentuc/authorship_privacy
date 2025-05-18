# authorship_privacy
This is implementation of the paper: Unraveling the Interwoven Roles of Large Language Models in Authorship Privacy: Verification, Obfuscation, and Mimicking

## Installation
```bash
conda create --name AA python=3.11
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install conda-forge::openai
conda install conda-forge::tiktoken
conda install matplotlib
pip install datasets
pip3 install torch transformers pandas
pip3 install peft bitsandbytes accelerate
conda install -c conda-forge tensorflow-hub
pip install tf-keras
pip install gensim
```

## Data Preprocessing
Quora: 
- Randomly get 200 authors who have: profile and at least 50 writings (done)
- Randomly get 50 samples and split train/val/test=40/5/5 (done)
- Define the template for the user_profile (done)
- Use ChatGPT API to generate final user_profile with some specific attribute (done)
- Run chatGPT to get synthesize dataset (let it run on the 40 train sample of train set) (running)
- Evaluate on the dataset and synthesize dataset with bert, n-gram (monday)