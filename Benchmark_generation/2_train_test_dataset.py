import os
import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import shutil

# get the list of 200 authors who have both writing and profile for training
profile_path ='/media/volume/arkai-lab-data-private/Coding/AA/Datasets/Quora/quora_merge_profile/'
quora_writing = '/media/volume/arkai-lab-data-private/Coding/AA/Datasets/Quora/raw_users_writing/'
bechmark_dataset = '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/'

# filterout the good user and their writing
def get_benchmark_dataset():
    # get the list of valid profile
    valid_user_profile = []
    for file_name in os.listdir(profile_path):
        file_path = os.path.join(profile_path, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if len(data.keys()) >=7:
                if len(data['credential']) >=9:
                    valid_user_profile.append(file_name)

    # randomly choosing author from the list of valid_user_profile
    # check number of writing
    print("Valid:", len(valid_user_profile))
    i = 0
    good_user_writing ={}
    while i <= 200:
        if len(valid_user_profile) == 0:
            break
        random_user = random.choice(valid_user_profile)
        valid_user_profile.remove(random_user)
        writing_path = quora_writing+ random_user.split('.')[0]+'.csv'
        writing = pd.read_csv(writing_path)
        # each randomuser, check the number of writing more than 50
        # each document, the length of writing greater than 50
        if writing.shape[0] > 50:
            # select randomly writing of author
            all_records = []
            num_record_leng_more_than200 = 0
            for index, row in writing.iterrows():
                if len(str(row['Answer'])) > 50:
                    all_records.append([row['Name'], row['Question'], row['Answer'], row['Image']])
                    num_record_leng_more_than200 += 1
            print("Number record more than 200:", num_record_leng_more_than200)
            if num_record_leng_more_than200 >= 50:
                i+=1
                pandas_record = pd.DataFrame(all_records, columns=['Name','Question','Answer','Image'])
                print("Writing")
                pandas_record.to_csv(bechmark_dataset+ 'all/'+ random_user.split('.')[0]+'.csv', index=False)
        print("Number of valid:",i)

def split_train_eval_test(data_path):
    file_name = os.listdir(data_path)
    for f in file_name:
        print("User: ", f)
        user_writing_path = data_path+f
        user_writing = pd.read_csv(user_writing_path)
        randomly_sample_50 = user_writing.sample(n=50, random_state=2024, replace=False)
        # split into train/validation/test=40/5/5
        train_df, test_df = train_test_split(randomly_sample_50, test_size=0.1, random_state=2024)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2024)
        # save to train/val/test
        train_df.to_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/train/'+f, columns=['Name','Question','Answer','Image'], index=False)
        val_df.to_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/eval/'+f, columns=['Name','Question','Answer','Image'], index=False)
        test_df.to_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/test/'+f, columns=['Name','Question','Answer','Image'], index=False)

def get_10_more_writing_each_writers(data_path):
    file_name = os.listdir(data_path)
    for f in file_name:
        if f in os.listdir('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/10_more_original_writing/'):
            # print("Skip:", f)
            continue

        user_writing_path = data_path+f
        user_writing = pd.read_csv(user_writing_path)
        old_train_val_test_50 = user_writing.sample(n=50, random_state=2024, replace=False)
        print("Old index:", list(old_train_val_test_50.index))

        print("Shape:", user_writing.shape)
        if user_writing.shape[0] == 50:
            print("No more to take")
            continue

        # sampling more 10 for augmentation
        new_10 = []
        stop_cond = False
        while not stop_cond:
            new = user_writing.sample(n=1, random_state=2024, replace=False)
            if new.index in list(old_train_val_test_50.index):
                continue
            new_10.append(new)
            if len(new_10) == 10:
                stop_cond = True

        new_10_pd = pd.concat(new_10, ignore_index=True)
        print(new_10_pd.shape[0])
        new_10_pd.to_csv('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/10_more_original_writing/'+f, columns=['Name','Question','Answer','Image'], index=False)

def get_user_profile(data_path):
    # get list of uid
    user_list = []
    user_ids = os.listdir(data_path)
    for uid in user_ids:
        user_list.append(uid.split('.')[0])

    # copy and paste file
    for uid in user_list:
        print("Copying: ", uid)
        shutil.copy('/media/volume/arkai-lab-data-private/Coding/AA/Datasets/Quora/quora_merge_profile/'+uid+'.json', '/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/user_profile/'+uid+'.json')
# get_benchmark_dataset()
# split_train_eval_test('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/all/')
# get_user_profile('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/all/')
# get 10 more writing for each
# get_10_more_writing_each_writers('/media/volume/arkai-lab-data-private/Coding/AA/Benchmark_generation/quora/all/')