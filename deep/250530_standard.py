from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# randn() -> 평균이 0이고, 표준편차가 1인 정규분포 난수 생성기
# # values = [ np.random.randn() for _ in range(10000)]
# 곱하기는 표준편차를 조정하고, 더하기는 평균을 조정한다.
# randn()에 난수 발생 범위가 없는 이유는 (평균 0, 표준편차 1인)정규분포이기 때문이다.

# plt.hist(values, bins=30, color="purple",edgecolor='black')
# plt.show()

scaler = StandardScaler()

values = np.arange(10).reshape(-1, 1)

# fit() 메서드는 현재 데이터의 평균, 분산, 표준편차를 계산한다.
fit_values = scaler.fit(values)

print(fit_values.mean_, fit_values.var_, fit_values.scale_)
# transform() 메서드는 데이터를 표준화한다.
# fit_transform() 메서드는 fit()과 transform()을 동시에 수행한다. 
fit_values = scaler.fit_transform(values)
print(fit_values)

# inverse_transform() 메서드는 표준화된 데이터를 원래의 스케일로 되돌린다.
inverse_values = scaler.inverse_transform(fit_values)
print(inverse_values)
