# data_loader.py

#fp1 2022.01.01~2022.06.30
#fp2 2022.07.01~2022.12.31
#fp3 2023.01.01~2023.06.30

import pandas as pd

def load_excel_data(file_path, sheet_dict):
    data = {}
    for sheet_name, sheet_index in sheet_dict.items():
        data[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_index)
    return data

def load_all_data():
    # 파일 경로 및 시트 정보 정의
    file_paths = {
        'data1': '/home/nayoung/Blackboard/data/data1.xlsx',
        'data2': '/home/nayoung/Blackboard/data/data2.xlsx',
        'data3': '/home/nayoung/Blackboard/data/data3.xlsx',
    }

    sheet_info = {
        "UserChat": 0,
        "User": 1,
        "SupportBot": 2,
        "Manager": 3,
        "Bot": 4,
        "Message": 5,
        "UserChatTag": 6,
        "UserChatMeet": 7
    }

    # 모든 데이터셋 로드
    all_data = {}
    for data_key, file_path in file_paths.items():
        all_data[data_key] = load_excel_data(file_path, sheet_info)
    return all_data
