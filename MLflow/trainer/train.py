import mlflow

# docker-compose 내부 네트워크로 서버 연결
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage0-youngmin")

with mlflow.start_run():
    mlflow.log_param("test_param", 111)
    mlflow.log_metric("test_metric", 0.1234)
    
    print("MLflow 첫 실험기록!")