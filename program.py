import pandas as pd
from collections import Counter

# CSV 파일 로드
data = pd.read_csv('/aimlk/Blackboard/Analysis_data/user_intent_clusters.csv')

# 클러스터 데이터를 계층적으로 분리
data['cluster_levels'] = data['cluster'].str.split('.')

def get_cluster_descriptions(data, level, parent_cluster=''):
    # 해당 레벨의 클러스터 설명을 가져옴
    if parent_cluster:
        filtered_data = data[data['cluster'].str.startswith(parent_cluster)]
    else:
        filtered_data = data

    clusters = filtered_data['cluster_levels'].str[level - 1].unique()
    descriptions = {}
    for cluster in clusters:
        # 특정 클러스터에 해당하는 텍스트 데이터 필터링
        cluster_texts = filtered_data[filtered_data['cluster_levels'].str[level - 1] == cluster]['text']
        # 가장 흔한 단어 또는 문구 찾기
        most_common_text = Counter(cluster_texts).most_common(1)[0][0]
        descriptions[cluster] = most_common_text

    return descriptions

def display_clusters(level, parent_cluster=''):
    descriptions = get_cluster_descriptions(data, level, parent_cluster)
    for cluster, description in descriptions.items():
        print(f"클러스터 {cluster}: {description}")

# 메인 프로그램 루프
current_level = 1
current_cluster = ''
max_level = data['cluster_levels'].apply(len).max()

while True:
    display_clusters(current_level, current_cluster)
    user_input = input(f"다음 레벨의 클러스터를 보려면 클러스터 번호를 입력하세요 (종료하려면 'exit' 입력): ")

    if user_input.lower() == 'exit':
        break

    if not user_input.isdigit() or int(user_input) >= max_level:
        print("잘못된 입력입니다. 다시 시도하세요.")
        continue

    current_cluster += user_input + '.'
    current_level += 1

    if current_level > max_level:
        print("더 이상의 하위 레벨이 없습니다. 프로그램을 종료합니다.")
        break