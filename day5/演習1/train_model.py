import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# データの読み込み
data = pd.read_csv('data/test_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflowの設定
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("model_training")

# モデルの学習と保存
with mlflow.start_run():
    # モデルの学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # モデルの評価
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # メトリクスの記録
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)
    
    # モデルの保存
    mlflow.sklearn.log_model(model, "model")
    
    # モデルをローカルに保存
    os.makedirs('models', exist_ok=True)
    mlflow.sklearn.save_model(model, 'models/model.pkl')

print("モデルの学習と保存が完了しました。") 