import pandas as pd
import numpy as np
import json
from preprocess import preprocess
import datetime

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

# モデル学習
model = Classifier()
model.fit(X, y)

# 評価データの読み込み
df = pd.read_csv("./test.csv")

# データ前処理
preprocess(df)

# データの作成
X = df[use_columns]

# ラベル推定
df["Survived"] = model.predict(X)

# 結果書き込み
df.to_csv(f"./submittion_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.csv", columns=["PassengerId", "Survived"], index=False)
