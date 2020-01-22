# imports
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Data is from https://archive.ics.uci.edu/ml/datasets/Student+Performance
studentData = pd.read_csv("D:\pythonProj\LinearReg\student\student-mat.csv", sep=";")
# print(data.head())
studentData = studentData[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

# Extract features and labels
predict = "G3"
X = np.array(studentData.drop([predict], 1)) # Features
y = np.array(studentData[predict]) # Labels

# Split in to training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# get best model
# bestModel = 0
# for _ in range(20):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     # model definition
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accScore = linear.score(x_test, y_test)
#     print("Accuracy Score: " + str(accScore))
#
#     if accScore > bestModel:
#         bestModel = accScore
#         with open("studentGradesModel.pickle", "wb") as f:
#             pickle.dump(linear, f)

# Load best model
pickle_in = open("studentGradesModel.pickle", "rb")
linear = pickle.load(pickle_in)
y_pred= linear.predict(x_test)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
# The mean squared error
print('Mean squared error: \n', mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: \n', r2_score(y_test, y_pred))
print("-------------------------")

# testing some model predictions
# predicted= linear.predict(x_test)
# # for x in range(len(predicted)):
# #     print(predicted[x], x_test[x], y_test[x])

# Plotting model
# plot = "G2"
# plt.scatter(studentData[plot], studentData["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()
