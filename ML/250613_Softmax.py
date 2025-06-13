from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits()
# 2차원 벡터로 변환하는 이유는 각 이미지를 일렬로 나열하여 특징을 추출하기 위함
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

print(features[0])
print(labels[0]) # 0, 1, 2, ..., 9 , 0, 1, 2, ..., 9

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. 기본 설정
num_features = X_train_std.shape[1]
num_samples = X_train_std.shape[0] # 1437, 20%가 테스트셋으로 분할
num_classes = 10 # 0~9 클래스

W = np.random.randn(num_features, num_classes)
# (64, 10) 형태, 열의 개수가 분류해야하는 클래스 개수, 행의 개수가 특성의 개수
# 정규분포로 가중치 초기화, 왜 정규분포로 초기화하는가?
# 그 이유는 가중치가 너무 크거나 작으면 학습이 잘 되지 않기 때문

print(W)