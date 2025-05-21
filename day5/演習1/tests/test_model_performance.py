import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pickle
import os
from pathlib import Path
import json

class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        # 現在のディレクトリを取得
        self.current_dir = Path(__file__).parent
        
        # テストデータの読み込み
        data_path = self.current_dir.parent / 'data' / 'test_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Test data file not found at {data_path}")
        self.test_data = pd.read_csv(data_path)
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        
        # 現在のモデルの読み込み
        model_path = self.current_dir.parent / 'models' / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, 'rb') as f:
            self.current_model = pickle.load(f)
        
        # 過去のモデルの読み込み（存在する場合）
        old_model_path = self.current_dir.parent / 'models' / 'titanic_model.pkl'
        self.old_model = None
        if old_model_path.exists():
            with open(old_model_path, 'rb') as f:
                self.old_model = pickle.load(f)
        
        # 性能基準値の読み込み（存在する場合）
        baseline_path = self.current_dir.parent / 'models' / 'performance_baseline.json'
        self.baseline_metrics = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1': 0.82,
            'inference_time': 0.1  # 秒
        }
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                self.baseline_metrics = json.load(f)

    def test_inference_time(self):
        """推論時間が許容範囲内であることを確認"""
        # 10回の推論を実行して平均時間を計算
        inference_times = []
        for _ in range(10):
            start_time = time.time()
            _ = self.current_model.predict(self.X_test)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"\nAverage inference time: {avg_inference_time:.4f} seconds")
        self.assertLessEqual(avg_inference_time, self.baseline_metrics['inference_time'])

    def test_model_accuracy(self):
        """モデルの精度が基準値以上であることを確認"""
        y_pred = self.current_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nModel accuracy: {accuracy:.4f}")
        self.assertGreaterEqual(accuracy, self.baseline_metrics['accuracy'])

    def test_model_precision(self):
        """モデルの適合率が基準値以上であることを確認"""
        y_pred = self.current_model.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel precision: {precision:.4f}")
        self.assertGreaterEqual(precision, self.baseline_metrics['precision'])

    def test_model_recall(self):
        """モデルの再現率が基準値以上であることを確認"""
        y_pred = self.current_model.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel recall: {recall:.4f}")
        self.assertGreaterEqual(recall, self.baseline_metrics['recall'])

    def test_model_f1(self):
        """モデルのF1スコアが基準値以上であることを確認"""
        y_pred = self.current_model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f"\nModel F1 score: {f1:.4f}")
        self.assertGreaterEqual(f1, self.baseline_metrics['f1'])

    def test_model_consistency(self):
        """同じ入力に対する予測が一貫していることを確認"""
        # 同じデータで2回予測を実行
        pred1 = self.current_model.predict(self.X_test)
        pred2 = self.current_model.predict(self.X_test)
        np.testing.assert_array_equal(pred1, pred2)

    def test_performance_comparison(self):
        """過去のモデルと比較して性能が劣化していないことを確認"""
        if self.old_model is None:
            self.skipTest("No previous model available for comparison")
        
        # 現在のモデルの性能を計算
        current_pred = self.current_model.predict(self.X_test)
        current_accuracy = accuracy_score(self.y_test, current_pred)
        current_f1 = f1_score(self.y_test, current_pred, average='weighted')
        
        # 過去のモデルの性能を計算
        old_pred = self.old_model.predict(self.X_test)
        old_accuracy = accuracy_score(self.y_test, old_pred)
        old_f1 = f1_score(self.y_test, old_pred, average='weighted')
        
        print(f"\nPerformance comparison:")
        print(f"Current model - Accuracy: {current_accuracy:.4f}, F1: {current_f1:.4f}")
        print(f"Previous model - Accuracy: {old_accuracy:.4f}, F1: {old_f1:.4f}")
        
        # 性能が10%以上劣化していないことを確認
        self.assertGreaterEqual(current_accuracy, old_accuracy * 0.9)
        self.assertGreaterEqual(current_f1, old_f1 * 0.9)

if __name__ == '__main__':
    unittest.main() 