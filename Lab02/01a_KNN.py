import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=1/3, random_state=1)

print('Training size: %d' % len(y_train))
print('Test size: %d' % len(y_test))

for i in [1, 2, 4, 6, 8, 10]:
    print('k = %d\n' % i)
    model_KNN = KNeighborsClassifier(n_neighbors=i, p=2)
    model_KNN.fit(X_train, y_train)

    y_pred = model_KNN.predict(X_test)

    print('Accuracy_Score: %f' % accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=['SETOSA', 'VERSICOLR', 'VIRGINICA'], columns=['SETOSA', 'VERSICOLR', 'VIRGINICA'])
    print(cm_df)

    plt.figure()
    plt.plot(y_pred, label='Giá trị dự đoán (y_predicted)')
    plt.plot(y_test, label='Giá trị thực (y_test)')
    plt.xlabel('Mẫu số')
    plt.ylabel('Giá trị')
    plt.title('So sánh giữa giá trị thực và dự đoán')
    plt.legend()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion matrix')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.show()


