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
# H(x) = w * x + b
# randn -3 ~ 3까지의 난수 생성
y = 2.5 * X + np.random.randn(3, 1) * 2
y = y.ravel()

bar = np.random.rand(2, 3,2, 1)
print(bar)
print("--" * 10)
print(bar.ravel()) # n 차원 -> 1 차원으로


#--------------------------------------------------#

from sklearn.linear_model import SGDRegressor
# sklearn 여러가지 모델을 만들어서 제공을 함 

# 모델 생성 후 하이퍼파라미터 생성
model = SGDRegressor(# model 은 참조변수
    max_iter=100, # epoch수 - 이름은 모델마다 다르다
    learning_rate="constant", # 학습률 방법
    # constant = 상수 , invscaling = 점점 감소, adaptive = adam optimizer 다르게 적용
    eta0=0.001, # 왜 eta0 인가?
    penalty=None, # 정규화 없음
    random_state=0 
) 