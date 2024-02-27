import pandas as pd
import seaborn as sns
import numpy as np
import os
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm

font_path = Path(os.getcwd())/ "nanum_gothic/NanumGothic.ttf"
font = fm.FontProperties(fname=font_path, size=12)

# # fm.get_fontconfig_fonts()
# font_path = "/home/nayoung/.env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/NanumGothic.ttf"
# font_prop = fm.FontProperties(fname=font_path)

#fp1 2022.01.01~2022.06.30
#fp2 2022.07.01~2022.12.31
#fp3 2023.01.01~2023.06.30

# Load Excel file
fp1 = '/home/nayoung/Blackboard/data/data1.xlsx'
fp2 = '/home/nayoung/Blackboard/data/data2.xlsx'
fp3 = '/home/nayoung/Blackboard/data/data3.xlsx'

#----------------------

# Function to load data from all sheets of an Excel file
def load_all_sheets_data(fp):
    xl = pd.ExcelFile(fp)
    data_sheets = {}
    for i in range(len(xl.sheet_names)):
        data_sheets[xl.sheet_names[i]] = xl.parse(xl.sheet_names[i])
    return data_sheets

# Loading all sheets from each file
all_data1_sheets = load_all_sheets_data(fp1)
all_data2_sheets = load_all_sheets_data(fp2)
all_data3_sheets = load_all_sheets_data(fp3)

# Verifying by displaying the head of each sheet
all_data1_heads = {sheet: data.head() for sheet, data in all_data1_sheets.items()}
all_data2_heads = {sheet: data.head() for sheet, data in all_data2_sheets.items()}
all_data3_heads = {sheet: data.head() for sheet, data in all_data3_sheets.items()}

## Displaying heads of sheets from the first file as an example
# print(all_data1_heads)

#----------------------

# 컬럼 존재 여부 확인을 위한 코드 추가
# print("Columns in UserChat data:", all_data1_sheets['UserChat data'].columns)
# print("Columns in User data:", all_data1_sheets['User data'].columns)

# Checking columns of UserChat data and User data sheets in the first file to find a common column for merging
user_chat_columns_data1 = all_data1_sheets['UserChat data'].columns
user_data_columns_data1 = all_data1_sheets['User data'].columns

# Finding common columns between UserChat data and User data
common_columns_data1 = set(user_chat_columns_data1).intersection(set(user_data_columns_data1))

# Displaying the common columns
# print(common_columns_data1)

# Merging UserChat data and User data for each file and then merging the results from all three files
def merge_user_chat_and_user_data(user_chat_data, user_data):
    # Merging using an outer join on 'userId' column
    merged_data = pd.merge(user_chat_data, user_data, on='id', how='outer', suffixes=('_chat', '_user'))
    return merged_data

# Merging UserChat data and User data for each file
merged_data1 = merge_user_chat_and_user_data(all_data1_sheets['UserChat data'], all_data1_sheets['User data'])
merged_data2 = merge_user_chat_and_user_data(all_data2_sheets['UserChat data'], all_data2_sheets['User data'])
merged_data3 = merge_user_chat_and_user_data(all_data3_sheets['UserChat data'], all_data3_sheets['User data'])

# Merging the merged data from all three files
final_merged_data = pd.concat([merged_data1, merged_data2, merged_data3], ignore_index=True)

# # Displaying the head of the final merged data as a sample
# print(final_merged_data.head())

#----------------------
#----------------------시작
# Performing basic EDA on the merged dataset

# Function for basic exploratory data analysis on merged dataset
def basic_eda_merged_data(df):
    # Descriptive statistics

    # 누락된 값을 'Unknown'으로 대체
    df = df.fillna('Unknown')

    descriptive_stats = df.describe(include='all')

    # Plotting the distribution of a few selected columns as examples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribution of 'state' column
    sns.countplot(data=df, x='state', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Chat States')

    # Distribution of 'contactMediumType' column
    sns.countplot(data=df, x='contactMediumType', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Contact Medium Types')

    # Distribution of 'marketingType' column
    sns.countplot(data=df, x='marketingType', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Marketing Types')

    # Distribution of 'country' column in User data
    sns.countplot(data=df, x='country', ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of User Countries')

    plt.tight_layout()
    plt.show()

    return descriptive_stats

# Performing EDA on the merged dataset
eda_stats_merged_data = basic_eda_merged_data(final_merged_data)
eda_stats_merged_data.to_csv('/home/nayoung/Blackboard/Analysis_data/eda_stats_merged_data.csv', index=False)
print("eda_stats_merged_data:", eda_stats_merged_data)

#----------------------

# Additional analysis on the merged dataset

# Function for time-based analysis
def time_based_analysis(df):
    # Converting 'createdAt' to datetime and extracting hour for analysis
    df['createdAt_datetime'] = pd.to_datetime(df['createdAt_chat'])
    df['hour'] = df['createdAt_datetime'].dt.hour

    # Plotting distribution of chats over different hours of the day
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='hour')
    plt.title('Distribution of User Activity Over Hours of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Chats')
    plt.show()

print(time_based_analysis(final_merged_data))

#----------------------

# Function for inquiry analysis
def inquiry_analysis(df):
    if 'description' in df.columns:
        top_types = df['description'].value_counts().head(10)

        # Plotting top types of inquiry
        plt.figure(figsize=(12, 8))
        top_types.plot(kind='bar')
        plt.title('Inquiry Analysis', fontproperties=font)
        plt.xlabel('Type of Inquiry', fontproperties=font)
        plt.ylabel('Number of Inquiries', fontproperties=font)
        plt.xticks(fontproperties=font)

        plt.tight_layout()

        plt.show()
        return top_types
    else:
        return "No 'description' column found for inquiry analysis."

print(inquiry_analysis(final_merged_data))

#----------------------

# Selecting the appropriate column for user role
user_role_column = 'user_role' if 'user_role' in final_merged_data.columns else 'profile.user_role'

# Performing analysis on the distribution of user roles
user_role_distribution = final_merged_data[user_role_column].value_counts()

# Plotting the distribution of user roles
plt.figure(figsize=(12, 6))
user_role_distribution.plot(kind='bar')
plt.title('User Role Distribution Analysis', fontproperties=font)
plt.xlabel('User Role', fontproperties=font)
plt.ylabel('Number of Users', fontproperties=font)
plt.xticks(fontproperties=font)

plt.tight_layout()

plt.show()

print(user_role_distribution)

#----------------------

# Preprocessing the text data for analysis
# Function to clean and preprocess text data
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK 라이브러리 설정
nltk.download('punkt')
nltk.download('stopwords')

repetitive_sentence = ("건강장해 예방조치 사전 안내 산업안전보건법 제조고객의 폭언등으로 인한 "
                     "건강장해 예방조치에 의거하여 담당자를 감정노동에서 보호합니다 폭언을 삼가주시길 "
                     "정중히 요청드립니다 빠른 문제 해결을 위하여 정확한 정보를 제공해주시기를 요청드립니다 "
                     "시스템 특성 상 명확한 원인과 분석이 동반되어야 문제 해결이 가능하며 즉시 처리가 안될 "
                     "수 있는 부분 양해 부탁드립니다 로그인에 필요한 정보는 절대 알려주지 마세요 저희 팀원 "
                     "모두는 제기하신 민원에 대하여 최대 만족을 드릴 수 있도록 노력하겠습니다")

keywords = repetitive_sentence.split()

# 기본 텍스트 클리닝 함수
def clean_text(text):
    # 문장 단위로 분리
    sentences = text.split('\n')
    cleaned_sentences = []

    for sentence in sentences:
        # 키워드를 포함하는 문장 제거
        if not any(keyword in sentence for keyword in keywords) and sentence.strip() != '':
            cleaned_sentences.append(sentence)

    # 정제된 문장들을 다시 하나의 텍스트로 결합
    cleaned_text = ' '.join(cleaned_sentences)

    # 특수 문자 및 숫자 제거
    cleaned_text = re.sub(r'[^가-힣A-Za-z\s]', '', cleaned_text)
    return cleaned_text

# 추가적인 텍스트 전처리 함수
def preprocess(text):
    # 소문자 변환
    text = text.lower()
    # 토큰화
    tokens = word_tokenize(text)
    # 영어 불용어 제거 및 빈 토큰 제거
    tokens = [word for word in tokens if word not in stopwords.words('english') and word.strip() != '']
    return ' '.join(tokens)  # 토큰들을 다시 문자열로 결합

# 데이터 불러오기
message_data1 = all_data1_sheets['Message data']
message_data2 = all_data2_sheets['Message data']
message_data3 = all_data3_sheets['Message data']

# 데이터 병합
message_data = pd.concat([message_data1, message_data2, message_data3], ignore_index=True)

# 'plainText' 컬럼을 사용하여 텍스트 데이터 전처리
if 'plainText' in message_data.columns:
    # NaN 값을 None으로 변환 후 문자열로 변환
    message_data['cleaned_text'] = message_data['plainText'].replace({np.nan: None}).astype(str).apply(clean_text)
    message_data['processed_text'] = message_data['cleaned_text'].apply(preprocess)

    # 빈 문자열, 'nan', 'none'을 제거
    message_data['processed_text'] = message_data['processed_text'].apply(
        lambda x: x if x.strip() != '' and x.lower() != 'nan' and x.lower() != 'none' else None)

    # 'None' 값이 있는 행들 제거
    message_data.dropna(subset=['processed_text'], inplace=True)
else:
    print("'plainText' 컬럼이 데이터셋에 존재하지 않습니다.")

print(message_data['processed_text'])

#_____________________

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from kobert_transformers import get_tokenizer, get_kobert_model
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# NLTK VADER 감성 분석기 다운로드
nltk.download('vader_lexicon')

# KoBERT 모델 및 토크나이저 불러오기
model = BertForSequenceClassification.from_pretrained('monologg/kobert')
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

# 한국어 감성 분석 함수
def kobert_sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(1)
    # 긍정 클래스에 대한 확률 반환
    positive_prob = probs[:, 1].item()
    return positive_prob

# 기존의 sentiment_analysis 함수는 동일하게 유지
def sentiment_analysis(text):
    try:
        # 언어 감지
        language = detect(text)
        if language == 'en':
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)['compound']
        elif language == 'ko':
            # 한국어 감성 분석 로직 적용
            return kobert_sentiment_analysis(text)
        else:
            return None
    except Exception as e:
        return None

# Applying VADER sentiment analysis to the cleaned text data
message_data['sentiment'] = message_data['processed_text'].apply(sentiment_analysis)

# Displaying the distribution of sentiment
plt.figure(figsize=(10, 6))
sns.histplot(message_data['sentiment'], kde=True, bins=30)
plt.title('Sentiment Distribution in User Messages')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Displaying a few samples of the sentiment analysis
print(message_data[['processed_text', 'sentiment']])

message_data[['processed_text', 'sentiment']].to_csv('/home/nayoung/Blackboard/Analysis_data/sentiment_distribution.csv', index=False)

#----------------------

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import networkx as nx

# LDA 모델을 이용하여 클러스터의 토픽을 추출하는 함수
def extract_topics_for_cluster(texts, n_topics=1, n_words=10):
    if len(texts) == 0:  # 텍스트 데이터가 없는 경우
        return {}
    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, tokenizer=tokenize_korean_text)
        dtm = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names()
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            topics[f"Cluster {topic_idx} Topic"] = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        return topics
    except ValueError:
        return {}  # 어휘가 생성되지 않는 경우 빈 딕셔너리 반환


# Okt 형태소 분석기 초기화
okt = Okt()

# 한국어 텍스트를 형태소로 분리하는 함수
def tokenize_korean_text(text):
    pos_words = okt.pos(text, norm=True, stem=True)
    return [word for word, pos in pos_words if pos not in ['Josa', 'Conjunction']]

# TfidfVectorizer 설정
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_korean_text, stop_words= 'english' )
tfidf_matrix = tfidf_vectorizer.fit_transform(message_data['processed_text'].dropna())


def find_optimal_clusters(data, max_k):
    iters = range(2, min(max_k, data.shape[0]) + 1, 2)  # 데이터 포인트 수를 고려하여 max_k 설정

    best_score = -1
    best_k = 2
    for k in iters:
        if data.shape[0] <= k:  # 데이터 포인트 수가 클러스터 수보다 작거나 같으면 건너뜁니다.
            break
        model = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        labels = model.fit_predict(data)

        # 라벨이 하나만 있을 경우 점수 계산을 건너뜁니다.
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


# Function to perform hierarchical clustering
def hierarchical_clustering(data, level, batch_size=100, prefix='', original_texts=None):
    if level == 0 or data.shape[0] == 0:  # 데이터가 없거나 더 이상 분할할 수 없는 경우
        return pd.DataFrame({'cluster': [prefix] * len(original_texts), 'text': original_texts})

    optimal_clusters = min(find_optimal_clusters(data, 10), data.shape[0])  # 클러스터 수와 데이터 수를 고려
    clustering = MiniBatchKMeans(n_clusters=optimal_clusters, batch_size=batch_size)
    try:
        labels = clustering.fit_predict(data)
    except ValueError:
        return pd.DataFrame({'cluster': [prefix] * len(original_texts), 'text': original_texts})  # 데이터가 없는 경우 예외 처리

    results = pd.DataFrame()
    for i in tqdm(range(optimal_clusters), desc=f"Level {level} clustering"):
        mask = labels == i
        cluster_label = f"{prefix}.{i}" if prefix else f"{i}"
        sub_data = data[mask]
        sub_texts = original_texts[mask] if original_texts is not None else None
        if sub_data.shape[0] > 0:  # 하위 클러스터에 대한 데이터가 있는 경우에만 재귀 호출
            sub_results = hierarchical_clustering(sub_data, level - 1, batch_size, cluster_label, sub_texts)
            results = pd.concat([results, sub_results])

    return results

def build_tree_structure(cluster_results):
    tree = {}
    for cluster in cluster_results['cluster']:
        parts = cluster.split('.')
        node = tree
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]
    return tree

def plot_tree(tree, parent_name, graph, top_words_dict):
    for k, v in tree.items():
        node_name = f"{parent_name}.{k}" if parent_name else k
        if parent_name:
            graph.add_edge(parent_name, node_name)
        if v:  # Sub-clusters exist
            plot_tree(v, node_name, graph, top_words_dict)
        else:  # Leaf node, add topics
            if node_name in top_words_dict:
                topics_str = ", ".join(top_words_dict[node_name])
                graph.nodes[node_name]['label'] = f"{node_name}\n({topics_str})"

def visualize_cluster_tree(cluster_results, top_words_dict):
    tree = build_tree_structure(cluster_results)
    graph = nx.DiGraph()
    plot_tree(tree, '', graph, top_words_dict)

    # 레이아웃 변경 및 라벨 크기 조절
    plt.figure(figsize=(40, 10))
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')  # 여기서 레이아웃 알고리즘 변경 가능
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw(graph, pos, with_labels=True, labels=labels, arrows=False, font_size=5, node_size=50)

    plt.savefig('/home/nayoung/Blackboard/Analysis_data/cluster_tree.png', format='png', dpi=300)  # 고해상도 설정
    plt.show()


# 원본 텍스트 배열 또는 리스트를 준비합니다.
original_texts = message_data['processed_text'].dropna().values

# 클러스터링 실행 + 레벨 조정하기
cluster_results = hierarchical_clustering(tfidf_matrix, 10, original_texts=original_texts)

# 각 클러스터에 대한 토픽 추출
top_words_per_topic = {}
for cluster in cluster_results['cluster'].unique():
    cluster_texts = cluster_results[cluster_results['cluster'] == cluster]['text']
    cluster_topics = extract_topics_for_cluster(cluster_texts)
    top_words_per_topic[cluster] = cluster_topics

# 여기에 각 클러스터에 대한 토픽을 출력하는 코드를 추가
for cluster, topics in top_words_per_topic.items():
    print(f"Cluster {cluster}:")
    for topic_num, words in topics.items():
        print(f"  {topic_num}: {' '.join(words)}")
    print()

# 트리 그래프 시각화에 사용할 토픽 단어 목록
top_words_dict = {cluster: " ".join(words) for cluster, words in top_words_per_topic.items()}

# 트리 그래프 시각화
visualize_cluster_tree(cluster_results, top_words_dict)


# Visualizing the hierarchical structure
plt.figure(figsize=(30, 6))
sns.countplot(data=cluster_results, x='cluster')
plt.title('Hierarchical Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Messages')
plt.xticks(rotation=45)
plt.show()

# Saving the clusters data
cluster_results.to_csv('/home/nayoung/Blackboard/Analysis_data/user_intent_clusters.csv', index=False)
