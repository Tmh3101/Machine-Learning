import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_predicted = gnb.predict(X_test)

score = accuracy_score(y_test, y_predicted)

print(score)