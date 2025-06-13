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
y_train = y_train.reshape(-1, 1) # y_train을 열 벡터로 변환
y_test = y_test.reshape(-1, 1) # y_test을 열 벡터로 변환

w = np.random.randn(X_train.shape[1], 1)
b = np.random.randn()
learning_rate = 0.01
epochs = 1000

# prediction
# z = w * x + b
z = X_train @ w + b
print(z.shape)

# sigmoid(z) -> 1 / (1 + e^(-z))
prediction = 1 / (1 + np.exp(-z))

for i in range(epochs):
    # Error
    error = prediction - y_train
    # print(prediction.shape)
    # print(y_train.shape)

    # print(z[:3,:])
    # print(prediction[:3,:])


    # Get grandient fow w and b
    graddient_w = X_train.T @ error / len(X_train)
    graddient_b = error.mean()
    # print(graddient_w.shape)

    # update parameters : w, b
    w -= learning_rate * graddient_w
    b -= learning_rate * graddient_b

    # Display the loss value of the current epoch
    if epochs % 100 == 0:
        loss = -y_train*np.log(prediction) - (1 - y_train)*np.log(1-prediction + 1e-10)
        print(loss.mean())
        
# w -> 30개의 b값

test_z = X_test @ w + b
test_prediction = 1 / (1 + np.exp(-test_z))
# print((test_prediction > 0.5).astype(int))
test_result = (test_prediction > 0.5).astype(int)

accuracy = ((test_result == y_test).astype(int))

print(f"Accuracy: {accuracy.mean():.4f}")
