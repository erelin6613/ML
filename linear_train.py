import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
#from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing, svm, linear_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

data = pd.read_csv('Admission_Predict.csv')
#print(data)
#print(data.describe())
#scaler = StandardScaler()
encode = LabelEncoder()
#print(data)

print(data.corr(method = 'pearson'))
X = np.array(data.drop(['Serial No.', 'Chance of Admit '], 1))
y = np.array(data['Chance of Admit '])
#X = preprocessing.scale(X)

#data = scaler.fit(data)

X_train = X[: -20]
y_train = y[: -20]
X_test = X[-20 :]
y_test = y[-20 :]

print(X_train)
print('*****')
print(y_train)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf = linear_model.SGDRegressor(alpha = 0.0001, tol = 0.0001)
#clf = linear_model.LinearRegression()
#clf = linear_model.Ridge(alpha = 0.001, normalize = False)
clf = svm.LinearSVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
#r2_confidence = r2_score(y_test, clf.predict(X_test))
print('confidence: ', confidence)
#print('confidence: ', r2_confidence)

print('Admit prediction: ', clf.predict(X_test))
plt.scatter(data['CGPA'], data['Chance of Admit '])
plt.show()