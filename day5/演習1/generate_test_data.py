import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# テストデータの生成
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# DataFrameに変換
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# テストデータを保存
df.to_csv('data/test_data.csv', index=False)
print("テストデータを生成して保存しました。") 