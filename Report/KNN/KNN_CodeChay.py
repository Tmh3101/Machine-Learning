import pandas as pd
import numpy as np


def readDataFromCSVFile():
    df = pd.read_csv('iris_data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


# Giả sử tham số truyền vào là hợp lệ
def normalizeData(data, inp):
    data = pd.concat([data, inp], ignore_index=True)
    cols = data.columns.tolist()
    for col in cols:
        minValue = data[col].min()
        maxValue = data[col].max()
        data.loc[:, col] = (data.loc[:, col] - minValue) / (maxValue - minValue)
    return data.iloc[:-1, :], data.iloc[-1, :]


# Giả sử tham số truyền vào có cùng so lượng và tên thuộc tính
def calculateEuclideanDistance(a, b):
    M = len(a)
    distance = 0
    for m in range(M):
        distance += a[m] - b[m]
    return distance


# chuyển từ list sang DataFrame
def convertToDataFrame(inp):
    return pd.DataFrame(np.array([inp]), columns=['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])


def KNeighborsNearestClassify(inp, k=5):
    X, y = readDataFromCSVFile()
    inp = convertToDataFrame(inp)
    X, inp = normalizeData(X, inp)

    # tính khoảng cách từ input đến các row của data và lưu vào một mảng
    distance = []
    for i in range(len(X)):
        distance.append(calculateEuclideanDistance(X.iloc[i, :], inp))
    distance = pd.DataFrame(distance, columns=['distance'])
    distance.sort_values(by='distance') # sắp xếp theo thứ tự tăng dần

    # lấy k khoảng cách nhỏ nhất
    k_neighbors_index = distance.iloc[:k, :].index.tolist()

    # tạo mảng unique của nhan và bộ đếm
    y_uni = y.unique()
    y_counter = [0] * len(y_uni)

    # duyệt qua k hàng xóm gần nhất -> kiểm tra và đếm các nhãn
    for i in k_neighbors_index:
        label = y.iloc[i]
        y_uni_index = np.where(y_uni == label)[0][0]
        y_counter[y_uni_index] += 1

    # trả về nhãn có số đếm lớn nhất
    return y_uni[y_counter.index(max(y_counter))]


inp = list()
inp.append(float(input('sepalLength: ')))
inp.append(float(input('sepalWidth: ')))
inp.append(float(input('petalLength: ')))
inp.append(float(input('petalWidth: ')))

res = KNeighborsNearestClassify(inp)
print('Kết quả dự đoán: ', res)


