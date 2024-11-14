import pandas as pd
import numpy as np

# Doc du lieu và xuat ra thong tin
data_frame = pd.read_csv("iris_data.csv")
print(data_frame)

# Hien thi n dong du lieu dau tien
n = int(input("Nhap so dong dau tien: "))
print(data_frame.head(n))

# In ra so dong va so cot cua du lieu
rows, cols = data_frame.shape
print(f"So dong: {rows}")
print(f"So cot: {cols}")

# Lay cac thuoc tinh va nhan
X = data_frame.iloc[:, :-1]
Y = data_frame.iloc[:, -1]
print("Cac thuoc tinh:")
print(X)
print("Nhan:")
print(Y)
print()

# Lay thong tin thong ke du lieu o moi cot va ca tap dataset (su dung ham description())
print(data_frame.describe())


