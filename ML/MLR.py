import numpy as np

# 데이터 설정
np.random.seed(1)
num_samples = 5
num_features = 2

# X: 방 개수(x1), 전용 면적(x2) - 랜덤으로 0~10 사이 숫자
X = np.random.rand(num_samples, num_features) * 10

# 실제 집값을 만드는 진짜 w, b (모르지만 우리가 흉내낼 타깃)
true_w = np.array([5, 3])  # 방 개수는 집값에 5만 원 영향, 면적은 3만 원 영향
true_b = 4                 # 기본 집값 4만 원

# y: 실제 집값 (노이즈 포함)
noise = np.random.randn(num_samples)  # 약간의 랜덤 오차
y = X @ true_w + true_b + noise       # y = 5*x1 + 3*x2 + 4 + noise

# 초기 파라미터 설정
w = np.zeros(num_features)  # w = [0, 0]
b = 0.0
learning_rate = 0.01
epochs = 10

# 학습 시작
for epoch in range(epochs):
    # 1. 예측값 계산: h(x) = X @ w + b
    y_pred = X @ w + b

    # 2. 오차 계산
    error = y_pred - y

    # 3. 기울기 계산 (gradient)
    grad_w = (1 / num_samples) * (X.T @ error)  # 각 w의 변화량
    grad_b = (1 / num_samples) * np.sum(error)  # b의 변화량

    # 4. 파라미터 업데이트
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # 5. 손실 계산 (평균 제곱 오차)
    cost = (1 / (2 * num_samples)) * np.sum(error ** 2)

    # 6. 출력
    print(f"[{epoch+1} epoch] w = {w.round(2)}, b = {round(b, 2)}, cost = {round(cost, 2)}")
