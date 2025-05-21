import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import pickle
from pathlib import Path

# 現在のディレクトリを取得
current_dir = Path(__file__).parent

# データの読み込み
data_path = current_dir / 'data' / 'test_data.csv'
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)
X = data.drop('target', axis=1)
y = data['target']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflowの設定
mlruns_path = current_dir / 'mlruns'
mlflow.set_tracking_uri(f"file:{mlruns_path}")
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
    models_dir = current_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'model.pkl'
    
    # モデルをpickleで保存
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # MLflowにも保存
    mlflow.sklearn.log_model(model, "model")

print(f"モデルの学習と保存が完了しました。保存先: {model_path}") 