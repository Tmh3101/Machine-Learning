import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data từ file csv
data = pd.read_csv('iris_data.csv')

# Tách thuộc tính (X) và nhãn (y)
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Xử lý dữ liệu nhiễu hoặc thiếu => thay thế các giá trị đó bằng giá trị trung bình của cột chứa nó
# ==> Sử dụng khi cần thiết => đối với dataset này thì không cần
# attrs = X.columns.values
# for column in attrs:
#     X[column] = X[column].replace(0, np.NaN)
#     mean = X[column].mean()
#     X[column] = X[column].replace(np.NaN, mean)

# Chia ra các tập để train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuẩn hóa dữ liệu với MinMaxScaler [0->1]
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Tính số lượng hàng xóm cần xét
sqrt_y_test = math.sqrt(len(y_test))
k = int(sqrt_y_test - 1 if sqrt_y_test % 2 == 0 else sqrt_y_test)

# Classify
knn_classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
knn_classifier.fit(X_train, y_train)

# Test - dự đoán
y_predicted = knn_classifier.predict(X_test)

# Cho ra kết quả - confusion matrix
cm = confusion_matrix(y_test, y_predicted)
score = accuracy_score(y_test, y_predicted)

print(cm)
print(score)





