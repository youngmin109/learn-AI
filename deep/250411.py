import numpy as np

samples = []
y = []  # 정답

w = [0.2, 0.3]
b = 0.1

gradient_w = [0.0, 0.0]
gradient_b = 0.0

for dp in zip(samples, y):
    # 예측값
    predict_y = w * f + b
    
    # Error : 예측 값 - 정답
    error = predict_y - y
    
    # w의 기울기 : sum(Error * each f)/ 샘플의 개수
    gradient_w += error * f
    
    # b의 기울기 : sum(Error) / 샘플의 개수
    gradient_b += error
    
w = w - gradient_w / len(samples)
b = b - gradient_b / len(samples)