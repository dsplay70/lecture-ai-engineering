import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import mlflow
import os
from pathlib import Path

class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        # デバッグ情報の出力
        print("\nCurrent working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
        
        # テストデータの読み込み（パスを修正）
        current_dir = Path(__file__).parent
        data_path = current_dir.parent / 'data' / 'test_data.csv'
        print("\nLooking for test data at:", data_path)
        print("Data directory contents:", os.listdir(current_dir.parent / 'data'))
        
        if not data_path.exists():
            raise FileNotFoundError(f"Test data file not found at {data_path}")
            
        self.test_data = pd.read_csv(data_path)
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        
        # モデルの読み込み（パスを修正）
        model_path = current_dir.parent / 'models' / 'model.pkl'
        print("\nLooking for model at:", model_path)
        print("Models directory contents:", os.listdir(current_dir.parent / 'models'))
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = mlflow.sklearn.load_model(str(model_path))
        
        # 過去のモデルのパフォーマンス（ベースライン）
        self.baseline_metrics = {
            'accuracy': 0.85,  # 例として設定
            'precision': 0.83,
            'recall': 0.82,
            'f1': 0.82,
            'inference_time': 0.1  # 秒
        }

    def test_model_accuracy(self):
        """モデルの精度がベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nModel accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, self.baseline_metrics['accuracy'])

    def test_model_precision(self):
        """モデルの適合率がベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel precision: {precision}")
        self.assertGreaterEqual(precision, self.baseline_metrics['precision'])

    def test_model_recall(self):
        """モデルの再現率がベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel recall: {recall}")
        self.assertGreaterEqual(recall, self.baseline_metrics['recall'])

    def test_model_f1(self):
        """モデルのF1スコアがベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel F1 score: {f1}")
        self.assertGreaterEqual(f1, self.baseline_metrics['f1'])

    def test_inference_time(self):
        """推論時間が許容範囲内であることを確認"""
        start_time = time.time()
        _ = self.model.predict(self.X_test)
        inference_time = time.time() - start_time
        print(f"\nInference time: {inference_time}")
        self.assertLessEqual(inference_time, self.baseline_metrics['inference_time'])

    def test_model_consistency(self):
        """同じ入力に対する予測が一貫していることを確認"""
        # 同じデータで2回予測を実行
        pred1 = self.model.predict(self.X_test)
        pred2 = self.model.predict(self.X_test)
        np.testing.assert_array_equal(pred1, pred2)

if __name__ == '__main__':
    unittest.main() 