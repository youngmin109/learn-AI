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
scaler = StandardScaler()  # StandardScaler는 평균 0, 분산 1
X_train = scaler.fit_transform(X_train)  # 훈련 데이터에 대해 평균과 분산을 계산하고, 데이터를 변환
X_test = scaler.transform(X_test)  # 테스트 데이터는 훈련 데이터의 평균과 분산을 사용하여 변환

# 4. 초기 설정
num_features = X_train.shape[1]  # 특성 개수 X.shape = (569, 30)
epochs = 100000
learning_rate = 0.01    

# 가중치, 편향 초기화 (정규분포)
W = np.random.randn(num_features, 1)  # weight는 특성의 개수만큼
# 소프트 맥스 회귀에서는 가중치가 클래스의 개수만큼 필요하기 때문에
# shape를 (num_features, num_classes)로 설정
b = np.random.randn()  # 편향은 1개

# 정답 레이블 reshape: (n_samples, 1)로 변환
y_train = y_train.reshape(-1, 1)  # 행은 자동으로 계산

# 5. 경사하강법 학습 루프
for epoch in range(epochs):
    
    # 5-1. 시그모이드 함수
    
    # logit 계산
    z = X_train @ W + b # (569, 1) 형태
    # print(z[:5])
    # print("----------------")
    # 시그모이드 함수 적용
    predictions = 1 / (1 + np.exp(-z))  # (569, 1) 형태
    # print(predictions[:5])
    # print("----------------")
    
    # 5-2. 오차 계산
    errors = predictions - y_train  # (569, 1) 형태
    # print(errors[:5])
    # print("----------------")
    # print(y_train[:5])
    
    # 5-3. 그래디언트 계산
    grad_W = X_train.T @ errors / len(X_train)  # (30, 1) 형태
    grad_b = np.mean(errors)  # 스칼라 값
    
    # 5-4. 파라미터 업데이트
    W -= learning_rate * grad_W  # (30, 1) 형태
    b -= learning_rate * grad_b  # 스칼라 값
    
    # 5-5. 손실 함수 계산 (로그 손실)
    loss = -np.mean(y_train * np.log(predictions + 1e-15) +
                    (1 - y_train) * np.log(1 - predictions + 1e-15))
    
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    # 학습 상황 출력 (옵션 : 조절 가능)
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print(f"Weights: {W[:5].flatten()}") # flatten()으로 1차원 배열로 변환
        print(f"Bias: {b:.4f}") 

# 6. 테스트 세트에 대한 예측 및 정확도 평가
z_test = X_test @ W + b
# 시그모이드 함수 적용
y_prob_test = 1 / (1 + np.exp(-z_test))  # (n_samples, 1) 형태
# 예측값을 0 또는 1로 변환
y_pred_test = (y_prob_test >= 0.5).astype(int)  # 0.5 임계값으로 이진 분류

# 정확도 계산
accuracy = np.mean(y_pred_test.flatten() == y_test)  # (n_samples,)
print(f"Test Accuracy: {accuracy:.4f}")