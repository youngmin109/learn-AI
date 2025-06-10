from sklearn.model_selection import train_test_split
import numpy as np

# X : input
# y : output (lable) 
X = np.random.rand(10, 2) * 5
y = np.random.randint(0, 2, size=10)

X_train, X_test, y_train, y_test =  \
    train_test_split(X, y, test_size=0.2, random_state=42)
    # random_state -> 동일한 값으로 계속 실험, 설정하지 않으면 매번 다른 결과가 나옴

print(X)

print(f"X_train.shape: {X_train.shape}")
print(X_train)
print(f"X_test.shape: {X_test.shape}")
print(X_test)
            
#