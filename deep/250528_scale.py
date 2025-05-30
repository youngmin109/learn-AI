import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.arange(0, 10)

X_sum = sum(X)
X_avg = X_sum / len(X)

# 분산 구하기
variance = 0.0

for x in X:
    variance += (x - X_avg) ** 2
variance /= len(X)
print(variance)


# numpy를 사용하는 이유
np_avg = X.mean()
np_variance = np.var(X)
print(np_avg, np_variance)

# 표준편차 구하기 (바닐라, numpy)
std_dev = np.sqrt(variance)
np_std = X.std()
print(std_dev, np_std)

