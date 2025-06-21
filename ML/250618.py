from sklearn.datasets import load_digits # 손글씨 숫자 데이터셋 로딩
from sklearn.model_selection import train_test_split # 학습/테스트 셋 분할
from sklearn.preprocessing import StandardScaler # 이는 데이터표준화
import numpy as np # 수치 연산을 위한 라이브러리
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리

# 1. 데이터셋 로딩 및 분할
digits = load_digits() # digits이란 변수에 손글씨 숫자 데이터셋을 로드
features = digits.data                   # (1797, 64): 8x8 이미지 벡터
labels = digits.target                  # (1797,): 0~9 클래스 정수

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
# 표준화를 해주는 이뉴는 각 특성의 평균을 0, 분산을 1로 맞추어 특성간의 스케일을 맞추기 위함\
scaler = StandardScaler() # StandardScaler 객체 생성
X_train_std = scaler.fit_transform(X_train) # 학습 데이터 표준화 
X_test_std = scaler.transform(X_test) 
# 학습데이터에서 fit(계산)한 평균/표준편차를 테스트 데이터에 적용
# 그래야지 동일한 조건에서 예측이 가능

# 4. 기본 설정
num_features = X_train_std.shape[1] # 특성의 개수: 64
num_saples = X_train_std.shape[0] # 샘플의 개수: 1437
num_classes = 10 # 0~9 클래스
learning_rate = 0.01 # 학습률
epochs = 100000 # 학습 반복 횟수

W = np.random.randn(num_features, num_classes) 
# 행의 개수 = 특성의 개수, 열의 개수 = 클래스의 개수
# random.randn() 함수는 표준 정규분포에서 난수를 생성 -> -3 ~ 3 사이의 값 
# 가중치를 정규분포로 초기화하는 이유는 가중치가 너무 크거나 작으면 학습이 잘 되지 않기 때문
b = np.zeros(num_classes) # zeros() 함수는 0으로 초기화, 1차원 배열 생성

# 5. 학습 시작
for epoch in range(1):

    # 5-1. 로짓 계산
    # X (1437, 64) @ W (64, 10) + b (10,) = (1437, 10)
    logit = X_train_std @ W + b

    # logit_max를 구하는 이유는 소프트맥스 함수 계산(지수함수 계산 시) 시 오버플로우를 방지하기 위함
    # 컴퓨터가 표현할 수 있는 최대 수치 범위를 초과하는 값이 나올 때 발생하는 오류
    logit_max = np.max(logit, axis=1, keepdims=True)
    logit -= logit_max # (1437, 10) - (1437, 1) = (1437, 10) 
    # 차원을 유지하는 이유는 브로드 캐스팅 (서로 다른 차원끼리 연산할 수 있도록) 때문
    
    exp_logit = np.exp(logit) # 지수함수 적용 # logit값이 음수일 수도 있으니까 지수함수를 적용
    exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True) # 소프트맥스 함수의 분모 계산
    
    # 5-2 소프트맥스 함수 계산
    # 소프트맥스는 logit을 확률로 변환하는 함수
    softmax = exp_logit / exp_logit_sum 
    
    # 5-3. 원-핫 인코딩
    i_matrix = np.eye(num_classes) # (10, 10) 단위 행렬 생성
    one_hot = i_matrix[y_train] # numpy에서는 배열[인덱스]로 원-핫 인코딩 가능

    # error 계산
    error = softmax - one_hot #
    
    # 5-4. 경사 하강법
    # W와 b의 기울기 계산