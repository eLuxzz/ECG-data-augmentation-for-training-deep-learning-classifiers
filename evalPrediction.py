import numpy as np
import pandas as pd
from sklearn.metrics import classification_report





csv_data = pd.read_csv("data/PTB_XL_data/test_data.csv")
print(csv_data.head())  # Check first 5 rows
# print(csv_data.shape)
npy_data = np.load("dnn_predicts/base_predict.npy")  # Load the.npy file
# print(npy_data.shape)  # Check shape
print(npy_data[:5])
# print(classification_report(csv_data.values, npy_data))  # y_true is the ground truth labels