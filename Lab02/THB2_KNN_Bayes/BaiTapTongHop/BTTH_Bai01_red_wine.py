import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

print('----------Hold-Out----------')

df = pd.read_csv('winequality-red.csv', sep=';')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
print('Number of test elements: %d' % len(y_test))
print('Different label of elements:\n', y_test.unique())


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_knn = KNeighborsClassifier(n_neighbors=9)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)

print('KNN:')
print("First 7 elements' y_pred: " + str(y_pred_knn[:7]))
print("First 7 elements' y_test: " + str(y_test.values[:7]))
print(confusion_matrix(y_test, y_pred_knn))

model_Bayes = GaussianNB()
model_Bayes.fit(X_train, y_train)
y_pred_bayes = model_Bayes.predict(X_test)

print('Bayes:')
print("First 7 elements' y_pred: " + str(y_pred_bayes[:7]))
print("First 7 elements' y_test: " + str(y_test.values[:7]))
print(confusion_matrix(y_test, y_pred_bayes))

print("KNN Model's Accuracy Score: %f" % accuracy_score(y_test, y_pred_knn))
print("Bayes Model's Accuracy Score: %f" % accuracy_score(y_test, y_pred_bayes))

