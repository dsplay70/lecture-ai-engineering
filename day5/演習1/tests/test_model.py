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
        # テストデータの読み込み
        self.test_data = pd.read_csv('../data/test_data.csv')
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        
        # モデルの読み込み
        self.model_path = '../models/model.pkl'
        self.model = mlflow.sklearn.load_model(self.model_path)
        
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
        self.assertGreaterEqual(accuracy, self.baseline_metrics['accuracy'])

    def test_model_precision(self):
        """モデルの適合率がベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        self.assertGreaterEqual(precision, self.baseline_metrics['precision'])

    def test_model_recall(self):
        """モデルの再現率がベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred, average='weighted')
        self.assertGreaterEqual(recall, self.baseline_metrics['recall'])

    def test_model_f1(self):
        """モデルのF1スコアがベースライン以上であることを確認"""
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        self.assertGreaterEqual(f1, self.baseline_metrics['f1'])

    def test_inference_time(self):
        """推論時間が許容範囲内であることを確認"""
        start_time = time.time()
        _ = self.model.predict(self.X_test)
        inference_time = time.time() - start_time
        self.assertLessEqual(inference_time, self.baseline_metrics['inference_time'])

    def test_model_consistency(self):
        """同じ入力に対する予測が一貫していることを確認"""
        # 同じデータで2回予測を実行
        pred1 = self.model.predict(self.X_test)
        pred2 = self.model.predict(self.X_test)
        np.testing.assert_array_equal(pred1, pred2)

if __name__ == '__main__':
    unittest.main() 