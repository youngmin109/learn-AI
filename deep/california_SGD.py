# 필수 라이브러리 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split # 훈련/테스트 데이터 분할 함수
from sklearn.preprocessing import StandardScaler # 특정 정규화를 위한 스케일러
from sklearn.metrics import mean_squared_error, r2_score # 평가 지표
import matplotlib.pyplot as plt # 시각화

# 1. 데이터 로딩
# fetch_california_housing() 함수로 입력 데이터 X와 타겟값 Y를 불러옴
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. 학습용/테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 입력 특성 정규화 (표준화: 평균 0, 표준편차 1로 맞춤)
# SGD는 입력값 크기에 민감하므로 반드시 정규화가 필요
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 훈련 데이터를 기준으로 스케일 조정
X_test_scaled = scaler.transform(X_test) # 동일한 기준으로 테스트 데이터 변환

# 4. SGDRegressor 모델 정의 및 학습 (결과 값에 따라 튜닝)
model = SGDRegressor(
    max_iter=1000, # 최대 반복 횟수
    tol=0.01, # 손실이 tol보다 작아지면 학습 중단 (수렴 기준)
    eta0=0.001, # 학습률 (초기값)
    learning_rate='constant', # 학습률 조정 방법 (상수로 설정)
    penalty=None, # 정규화 없음 (과적합 방지 설정 사용 안함)
    random_state=42 # 결과 재현을 위한 시드 고정
)
model.fit(X_train_scaled, y_train) # 모델 학습

# 5. 예측 및 평가
y_pred = model.predict(X_test_scaled)
