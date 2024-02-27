# data_merge.py

from data_loader import load_all_data

import pandas as pd

all_data = load_all_data()

# 데이터 사용 예시
# print(all_data['data1']["UserChat"].head())

def merge_all_data(all_data):
    # Adjusting data types for merging
    # Converting 'id' columns to string type in all datasets to ensure compatibility
    datasets = [all_data['data1']["UserChat"], all_data['data1']["User"], all_data['data1']["SupportBot"],
                all_data['data1']["Manager"], all_data['data1']["Bot"], all_data['data1']["Message"],
                all_data['data1']["UserChatTag"]]

    for dataset in datasets:
        if 'id' in dataset.columns:
            dataset['id'] = dataset['id'].astype(str)

    # Reattempting the merge
    try:
        merged_data = pd.merge(datasets[0], datasets[1], on='id', how='outer', suffixes=('_userchat', '_user'))
        merged_data = pd.merge(merged_data, datasets[2], on='id', how='outer')
        merged_data = pd.merge(merged_data, datasets[3], on='id', how='outer', suffixes=('', '_manager'))
        merged_data = pd.merge(merged_data, datasets[4], on='id', how='outer', suffixes=('', '_bot'))
        merged_data = pd.merge(merged_data, datasets[5], left_on='id', right_on='chatId', how='outer')
        merged_data = pd.merge(merged_data, datasets[6], on='channelId', how='outer', suffixes=('', '_userchattag'))

    except Exception as e:
        print(f"Merge error: {e}")
        return None

    return merged_data
