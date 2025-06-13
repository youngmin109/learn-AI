import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지가 중요) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler() # StandardScaler는 평균 0, 분산 1로 변환
# 훈련 데이터에 대해 평균과 분산을 계산하고, 데이터를 변환한다.
X_train = scaler.fit_transform(X_train)
# 테스트 데이터는 훈련 데이터의 평균과 분산을 사용하여 변환한다.
X_test = scaler.transform(X_test)

# 4. 초기 설정
num_features = X_train.shape[1] # 특성 개수 X.shape = (569, 30)
epochs = 100000
learning_rate = 0.01

# 가중치, 편향 초기화 (정규분포)
# weight는 특성의 개수만큼, 편향은 1개
weights = np.random.randn(num_features, 1) 
# randn(a,b) -> a행 b열의 정규분포 난수 생성
bias = np.random.randn()

# 정답 레이블 reshape: (n_samples, 1)로 변환
y_train = y_train.reshape(-1, 1) # 행은 자동으로 계산, 1은 열의 개수

# 5. 경사하강법 학습 루프
for epoch in range(epochs):
    # 예측값 계산
    # 행렬 곱산으로 X_train (569, 30) 과 weights (30, 1) 곱셈
    # z는 (569, 1) 형태
    z = X_train @ weights + bias 
    
    
    predictions = 1 / (1 + np.exp(-z)) # np.exp()는 지수 함수 e^x 계산
    
    # 오차 계산
    # predictions (569, 1) - y_train (569, 1) -> (569, 1)
    errors = predictions - y_train
    
    # 그래디언트 계산
    # 
    grad_weights = X_train.T @ errors / len(X_train)
    grad_bias = np.mean(errors)
    
    # 파라미터 업데이트
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    
    # 손실 함수 계산 (로그 손실)
    loss = -np.mean(
        y_train * np.log(predictions + 1e-15) +
        (1 - y_train) * np.log(1 - predictions + 1e-15)
    )
    
    # 학습 상황 출력 (옵션 : 조절 가능)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        

# 6. 테스트 세트에 대한 예측 및 정확도 평가
z_test = X_test @ weights + bias
y_prob_test = 1 / (1 + np.exp(-z_test))
y_pred_test = (y_prob_test >= 0.5).astype(int)  # 0.5 임계값으로 이진 분류

# 정확도 평가
test_accuracy = np.mean(y_pred_test.reshape(-1) == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")