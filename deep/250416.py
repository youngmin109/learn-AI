import numpy as np

num_features = 3 # 특성 = 열 수
num_samples = 2 # 샘플 = 행 수수

# 난수 값 고정
np.random.seed(1)
np.set_printoptions(suppress=True, precision=3)
X = np.random.rand(num_samples, num_features)

# H(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_features)
b_true = np.random.randn() * 0.5

# 직접계산은 특성이 늘어날수록 길어진다. 행렬곱을 쓰자!
y = X[:,0] * w_true[0] + X[:,1] * w_true[1] + X[:,2] * w_true[2] + b_true

# 행렬 곱 (벡터 내적) + bias
y_ = X @ w_true + b_true
# numpy가 w_true의 열벡터를 행벡터로 변환 (연산자 오버라이딩)
# NumPy가 자동으로 적절한 내적을 수행
print(f"{y} \n {y_}")

w = np.random.rand(num_features)
b = np.random.randn()
learning_rate = 0.01
epochs = 10000

for epoch in range (epochs):
    # 예측 값
    prediction = X @ w + b 

    # error = 현재 샘플의 error값 (기울기 값을 구하기 위해 필요하다.)
    error = prediction - y
    # X가 샘플별에서 X.T는 특성별로 데이터를 모아준 형태가 된다.
    print(w)
    print(b)

    # gradient
    gradient = X.T @ error / num_samples

    w = w - learning_rate * gradient
    b = b - learning_rate * error.mean()
    print(w, b)