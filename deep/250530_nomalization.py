from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)

max = x.max()
min = x.min()

values = [ (item - min) / (max - min) for item in x]
print(values)

one_hot = np.eye(4)

print(one_hot, "\n\n")
# one-hot 인코딩된 배열에서 특정 인덱스의 행을 선택
# y_list는 선택할 행의 인덱스 리스트
y_list = [0, 1, 0, 3, 2, 3]
one_hot_value = one_hot[y_list]
print("one-hot 인코딩된 값:")
print(one_hot_value)

# one-hot 인코딩된 배열에서 각 행의 최대값 인덱스를 찾기
print("각 행의 최대값 인덱스:")
print(np.argmax(one_hot_value, axis=1))