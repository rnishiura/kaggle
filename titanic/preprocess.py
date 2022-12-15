import pandas as pd
import numpy as np

def preprocess(df):
  df['Sex'] = df['Sex'].replace(to_replace=['female', 'male'], value=[0, 1])
  df['Fare'] = df['Fare'].replace(to_replace=np.nan, value=5)
  df['Age'] = df['Age'].replace(to_replace=np.nan, value=25)

