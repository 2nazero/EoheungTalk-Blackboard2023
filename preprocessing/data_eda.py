#data_eda.py

import sys
# 모듈이 있는 디렉토리 경로 추가
sys.path.append('/Blackboard/preprocessing')

from data_merge import merge_all_data
from data_loader import load_all_data

# 데이터 병합 실행
all_data = load_all_data()
merged_data = merge_all_data(all_data)

# 병합된 데이터 사용
if merged_data is not None:
    print(merged_data.head())
else:
    print("Data merging failed.")

import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing: Handling missing values and outliers
# Calculating the percentage of missing values in each column
missing_values = merged_data.isnull().mean() * 100

# EDA: Basic statistical analysis
eda_summary = merged_data.describe()

# Correlation Analysis
# 선택적으로 수치형 데이터만 필터링
numeric_data = merged_data.select_dtypes(include=['float64', 'int64'])

# 상관관계 계산
correlation_matrix = numeric_data.corr()

# Visualizing the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

missing_values, eda_summary