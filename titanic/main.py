import pandas as pd
# from sklearn.svm import SVC as Classifier
from sklearn.ensemble import RandomForestClassifier as Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")
df = df.dropna()
df['Sex'] = df['Sex'].replace(to_replace=['female', 'male'], value=[0, 1])

df.info()



# X, y = df.loc[:,["Age"]], df.loc[:,"Survived"]
# X, y = df.loc[:,["Fare"]], df.loc[:,"Survived"]
X, y = df[["Age", "Fare", "Sex"]], df["Survived"]

# raise ValueError()

total_acc = 0
for i, (train_index, test_index) in enumerate(KFold(n_splits=5).split(X)):
  model = Classifier()

  X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred, normalize=False)
  total_acc += acc
  print(f"{acc} / {len(y_pred)} ({acc/len(y_pred)})")

print(f"{total_acc} / {len(y)} ({total_acc/len(y)})")

