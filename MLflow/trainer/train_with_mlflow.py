import mlflow
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from features import extract_features

# 서버 URI (docker-compose 네트워크 명)
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage1-noise-experiment_youngmin") # 실험 이름 설정

# 데이터 경로
base_dir = './data'
class_map = {'non_noisy': 0, 'noisy': 1}

# 데이터 로드
X, y = [], []
# 각 클래스 디렉토리에서 오디오 파일을 읽고 특징 추출
for class_name, label in class_map.items():
    class_dir = os.path.join(base_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.wav') or file_name.endswith('.mp3'): 
            file_path = os.path.join(class_dir, file_name) # 파일 경로
            features = extract_features(file_path) # 특징 추출 함수 호출
            X.append(features) # 특징 벡터
            y.append(label) # 클래스 레이블
            
X = np.array(X) 
y = np.array(y)

# 데이터 분할
# 80% 학습, 20% 테스트
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼 파라미터
C = 1.0 # 정규화 강도 

with mlflow.start_run(): # MLflow 실행 시작
    # 기록 시작
    mlflow.log_param("model_type", "Logistic Regression") # 모델 유형 
    mlflow.log_param("C", C) # 하이퍼 파라미터 
    
    # 모델 학습
    model = LogisticRegression(C=C, max_iter=1000) # 로지스틱 회귀 모델
    model.fit(X_train, y_train) # 모델 학습
    
    # 검증
    preds = model.predict(X_test) # 예측
    accuracy = accuracy_score(y_test, preds) # 정확도 계산
    
    # 기록
    mlflow.log_metric("accuracy", accuracy) # 정확도 기록
    
    # 아티팩트 저장
    np.save("model_weights.npy", model.coef_) # 모델 가중치 저장
    mlflow.log_artifact("model_weights.npy") # MLflow에 아티팩트로 저장
    
    print(f"Model trained and logged with accuracy: {accuracy:.4f}") # 결과 출력