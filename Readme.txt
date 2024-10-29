1. N-gram model with Logistic or Naive Bayes
- charac n-gram: no need to use SVD
- word n-gram: need to use SVD for dimension reduction 

2. Naive Bayes has very bad performance for a high label classification compare to Logistic Regression
- NB: 0.1708
- Logistic: 0.4979

3. Dataset preprocessing
- Quora: 
    + Randomly get 200 authors who have: profile and at least 50 writings (done)
    + Randomly get 50 samples and split train/val/test=40/5/5 (done)
    + Define the template for the user_profile (done)
    + Use ChatGPT API to generate final user_profile with some specific attribute (done)
    + Run chatGPT to get synthesize dataset (let it run on the 40 train sample of train set) (running)
    + Evaluate on the dataset and synthesize dataset with bert, n-gram (monday)

4. Conda create
- python=3.11
- conda install anaconda::pandas
- conda install anaconda::scikit-learn
- conda install conda-forge::openai
- conda install conda-forge::tiktoken
- conda install matplotlib