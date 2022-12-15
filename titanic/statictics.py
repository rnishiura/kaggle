import pandas as pd
import numpy as np
from preprocess import preprocess

# データの読み込み
test_data = pd.read_csv("./test.csv")
train_data = pd.read_csv("./train.csv")

# 評価データへのSurvived列の追加
test_data['Survived'] = np.nan

# データ結合
df = pd.concat([train_data, test_data], ignore_index=True, sort=False)

# 情報表示
df.info()

# データ前処理
preprocess(df)

# 情報表示
df.info()