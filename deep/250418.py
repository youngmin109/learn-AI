import numpy as np

num_features = 4
num_samples = 10000
learning_rate = 0.01

np.random.seed(1)
np.set_printoptions(suppress=True, precision=1)

X = np.random.rand(num_samples, num_features) * 2
w_true = np.random.randint(1, 11, (num_features, 1)) # 매트릭스로 만들어버린다. 그러는 이유?
b_true = np.random.randn() * 0.5

y = X @ w_true + b_true

####################################################

# 초기의 w값은 달라야된다 -> ones X 랜덤값으로 만든다.
w = np.random.rand(num_features, 1)
b = np.random.rand()

gradient = np.zeros((num_features, 1)) # 기울기 초기화
# 예측 값

for _ in range(1000):
    predict_y = X @ w 

    # error
    error = predict_y - y

    # 기울기 계산 cost(xi) = error * xi
    gradient_x =  X.T @ error / num_samples
    gradient_b = error.mean()

    # 기울기 업데이트
    w = w - learning_rate * gradient_x
    b = b - learning_rate * gradient_b

print(f"w : {w.T} \n b : {b}")
print(f"w_true : {w_true.T} \n b_true : {b_true}")
print(f"error : {error.mean()}")
