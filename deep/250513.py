import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
# stratify=y: 클래스 비율을 유지하며 훈련/테스트 데이터 분할

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
learning_rate = 0.01

num_features = X_train.shape[1] # 특성 개수 30

w = np.random.randn(num_features, 1)
b = np.random.randn()
np.set_printoptions(suppress=True, precision=5) # 과학적 표기법 비활성화

y_train = y_train.reshape(-1, 1) # y_train을 열 벡터로 변환
# z = wx + b
z = X_train @ w + b

# prediction = 1 / (1 + e^(-z)) -> sigmoid
prediction = 1 / (1 + np.exp(-z))

print(prediction)
print(prediction.shape)

# error = prediction - y_train
error = prediction - y_train

# gradient_w, gradient_b
gradient_w = X_train.T @ error / len(X_train)
gradient_b = error.mean()

print(gradient_w)
print(gradient_w.shape)
# update parameters : w, b
w = w - learning_rate * gradient_w
b = b - learning_rate * gradient_b

# calculate loss
loss = -np.mean(y_train*np.log(prediction) + (1 - y_train)*np.log(1 - prediction))
print(f"loss: {loss:.4f}")