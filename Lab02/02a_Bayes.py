import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

model_Bayes = GaussianNB()
model_Bayes.fit(X_train, y_train)

y_pred = model_Bayes.predict(X_test)
print(y_pred)
print(y_test)
print('Accuracy_Score: %f' % accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
cm_df = pd.DataFrame(cm, index=['SETOSA', 'VERSICOLR', 'VIRGINICA'], columns=['SETOSA', 'VERSICOLR', 'VIRGINICA'])
print(cm_df)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion matrix')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()