import json
import os
# json profile of user crawl from quora
root_path = '/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/user_profile_json/users_profile_from_quora/'
linkedin_path = '/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/user_profile_json/user_profie_from_linkedin/user_logs_json/quora/'
for user_file in os.listdir(root_path):
    with open(root_path+ user_file, 'r') as f1:
        data1 = json.load(f1)
    
    # load profile from linkedin
    if user_file in os.listdir(linkedin_path):
        with open(linkedin_path+ user_file, 'r') as f2:
            data2 = json.load(f2)
            for k2, v2 in data2.items():
                if k2 not in data1.keys():
                    data1[k2] = data2[k2]
            # Save the merged data to a new JSON file
            with open('/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/user_profile_json/quora_merge_profile/'+user_file, 'w') as f_out:
                json.dump(data1, f_out, indent=4)
    else:
        # Save the merged data to a new JSON file
        with open('/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/user_profile_json/quora_merge_profile/'+user_file, 'w') as f_out:
            json.dump(data1, f_out, indent=4)

