from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits() # digits이란 변수에 손글씨 숫자 데이터셋을 로드
# 2차원 벡터로 변환하는 이유는 각 이미지를 일렬로 나열하여 특징을 추출하기 위함
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

# print(features[0]) 
# print(labels[0]) # 0, 1, 2, ..., 9 , 0, 1, 2, ..., 9

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
learning_rate = 0.01
epochs = 100000

W = np.random.randn(num_features, num_classes) # (64, 10)
    # (64, 10) 형태, 열의 개수가 분류해야하는 클래스 개수, 행의 개수가 특성의 개수
    # 정규분포로 가중치 초기화, 왜 정규분포로 초기화하는가?
    # 그 이유는 가중치가 너무 크거나 작으면 학습이 잘 되지 않기 때문
b = np.zeros(num_classes) # (10,)
    
for epoch in range(epochs): 

    # X (1437, 64) @ W (64, 10) + b (10,) = (1437, 10)
    logit = X_train_std @ W + b 
    # (1437, 10)이 나오는 이유는 각 샘플에 대해 10개의 클래스에 대한 점수를 계산하기 때문

    # 소프트맥스 함수 계산 시 오버플로우를 방지하기 위함
    logit_max = np.max(logit, axis=1, keepdims=True)
    # axis=1: 각 행의 최대값을 구함, keepdims=True: 결과를 1차원 상태가 아닌 2차원으로 유지
    # 차원을 유지하는 이유는 (1437,) 형태가 아닌 (1437, 1) 형태로 유지하기 위함

    logit -= logit_max # 각 행에서 최대값을 빼줌으로써 오버플로우 방지 
    # 나머지 값을 음수로 만들면 0으로 수렴하게 된다.

    exp_logit = np.exp(logit) #  지수함수 적용
    # np.exp()에서 exp는 자연상수 e의 거듭제곱을 의미
    exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True) # 소프트맥수 함수의 분모 계산
    # axis를 1로 설정하여 왼쪽에서 오른쪽으로 계산하는 행 중심 계산

    # print(np.sum(exp_logit_sum[0]))
    # print(np.sum(exp_logit_sum[10]))

    softmax = exp_logit / exp_logit_sum

    # print(softmax[0]) # 첫 번째 샘플의 소프트맥스 확률
    # print(np.sum(softmax[0])) 
    # print(np.sum(softmax[10]))
    # print(y_train[0])

    ## 다음으로 해야할 일은
    # 정답 레이블 y_train을 one-hot encoding으로 바꾸고
    # Cross Entropy Loss 계산하고
    # 경사 하강법으로 W, b를 업데이트해서
    # 점점 softmax[i][y_train[i]]가 높아지도록 만드는 것

    i_matix = np.eye(num_classes) # (10, 10) 단위 행렬
    one_hot = i_matix[y_train] # (1437, 10) 원-핫 인코딩

    ## print(one_hot[0]) # 첫 번째 샘플의 원-핫 인코딩 
    # error = softmax(1437, 10) - one_hot(1437, 10)

    error = softmax - one_hot # (1437, 10) 형태의 오차 행렬
    # 각 샘플에 대해 정답 클래스의 확률을 1로 만들고
    # 나머지 클래스의 확률을 0으로 만드는 것이 목표
    # 오차 행렬을 통해 가중치와 편향을 업데이트할 수 있다. 

    gradient_w = X_train_std.T @ error / num_samples
    gradient_b = np.mean(error, axis=0)

    
    # 가중치와 편향 업데이트
    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # loss
    loss = -np.log(softmax + 1e-15)* one_hot
    # 1e-15는 로그 계산 시 0으로 나누는 것을 방지

    if epoch % 1000 == 0:
        print(f"Cross Entropy Loss: {np.mean(loss)}") 
    
# 예측 모델
def predict(arg_X, arg_label):
    arg_X = scaler.transform(arg_X)
    logit = arg_X @ W + b
    logit -= np.max(logit, axis=1, keepdims=True)
    exp_logit = np.exp(logit)
    exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True)
    softmax = exp_logit / exp_logit_sum
    return np.argmax(softmax, axis=1)

# 기말고사 문제에 이 코드가 나온다.
# 코드를 설명할 수 있어야한다.