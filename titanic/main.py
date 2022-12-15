import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from preprocess import preprocess

# 学習モデル読み込み
# from sklearn.svm import SVC as Classifier
from sklearn.ensemble import RandomForestClassifier as Classifier

# 設定読み込み
with open('settings.json', 'r') as f:
  settings = json.load(f)

# 使用するカラムの設定
use_columns = settings['use_columns']

# 訓練データの読み込み
df = pd.read_csv("./train.csv")

# データ前処理
preprocess(df)

# データとラベルの分離
X, y = df[use_columns], df["Survived"]

# K分割交差検証
total_acc = 0
for i, (train_index, test_index) in enumerate(KFold(n_splits=10).split(X)):
  model = Classifier()

  X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred, normalize=False)
  total_acc += acc
  print(f"{acc} / {len(y_pred)} ({acc/len(y_pred)})")

# 総合評価
print(f"{total_acc} / {len(y)} ({total_acc/len(y)})")

