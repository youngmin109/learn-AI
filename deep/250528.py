import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.arange(0, 10)

print(sum(X))
avg = sum(X)/len(X)

deviation = 0
for item in X:
    deviation += item - avg
    
print(deviation)