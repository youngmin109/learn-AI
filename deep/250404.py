import numpy as np

# np.zeros() : numpy배열을 0으로 초기화해서 만드는 함수
foo = np.zeros((3, 2))
pos = np. zeros((2, 3, 2))

print(foo)
print(pos)

print(f"bar.shape: {foo.shape}")
print(f"bar.shape: {pos.shape}")

# random 0이상 1미만의 난수 생성성
print(type(np.random.rand(2,3)))

# tip 소수점 자름
np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(3, 1) * 10

pos = 2.5 * X
bar = np.random.randn(3, 1) * 2
# H(x) = w * x + b
# randn -3 ~ 3까지의 난수 생성
y = 2.5 * X + bar

print(X)
print("--" * 10)
print(pos)
print("--"* 10)
print(bar)
print("--" * 10)
print(y)
print("--" * 10)