import numpy as np

num_of_samples = 5
num_of_features = 2

# data set
np.random.seed(42)
# np.set_printoptions(suppress=True, precision=2)
np.set_printoptions(False, suppress=True)
''' np.set_printoptions(
    precision=None,       # 소수점 이하 자릿수
    suppress=False,       # 지수 표기 억제
    threshold=None,       # 생략 없이 출력할 최대 원소 수
    linewidth=None,       # 한 줄에 출력할 너비
    edgeitems=None,       # 앞뒤로 몇 개 항목을 보여줄지
    등등
)'''
X = np.random.rand(num_of_samples, num_of_features) * 10

x_true = [5, 3]
b_true = 4
noise = np.random.randn(num_of_samples) * 0.5
y = (X[:, 0] * 5 + X[:, 1] * 3 + b_true + noise)
print(X)
print(y)