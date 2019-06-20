import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing, svm, linear_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.cluster import KMeans
#from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split
#from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt




def category_conversion(frame):
	columns = frame.columns.values

	for col in columns:
		text_values = {}
		def convert_to_int(val):
			return text_values[val]
		#if frame[col]dtype != np.int64 or frame[col]dtype != np.float64:
		#	content = frame[col].tolist()



data = pd.read_csv('train.csv')
#print(data)
#plt.scatter(data['Fare'], data['Survived'])
#plt.show()

data['Age'].fillna(data['Age'].mean(), inplace = True)
data['Cabin'].fillna('Unknown', inplace = True)
data['Embarked'].fillna('S', inplace = True)

print(data.describe())
#scaler = StandardScaler()
encode = OrdinalEncoder()
#print(data)

X = np.array(data.drop(['PassengerId', 'Survived', 'Name'], 1))
y = np.array(data['Survived'])
print(np.shape(X))
#print(X[:, 9])
X = encode.fit_transform(X)
X = preprocessing.scale(X)
print(X[0])
print('****')
print(data.corr(method = 'pearson'))

X_train = X[: -20]
#y_train = y[: -20]
X_test = X[-20 :]
#y_test = y[-20 :]

clf = KMeans(n_clusters = 2)
#print(clf.cluster_centers_)
#clf = svm.SVC(kernel = 'linear', C = 1.0, gamma = 10)
y = clf.fit_predict(X_train)

#clf.cluster_centers_[0]
#or i in range:
#	plt.plot(clf.cluster_centers_[0, i], clf.cluster_centers_[1, i])
#for i in clf.cluster_centers_:
#	plt.plot(clf.cluster_centers_[: , 0], clf.cluster_centers_[0, :])
	#print(clf.cluster_centers_[i], data.loc['Survived'][i])
	#plt.scatter(center[])
plt.scatter(X_train, X_test)
plt.show()
for val in range(len(X_test)):
	print(X[:][val], 'will (s)he survive?', y[val])

#print(clf.cluster_centers_())
confidence = clf.score(X_test)
print('confidence: ', confidence)

#plt.scatter(data['Fare'], data['Survived'])
#plt.show()












#X = np.array(data.drop(['Serial No.', 'Chance of Admit '], 1))
#y = np.array(data['Chance of Admit '])
#X = preprocessing.scale(X)

#data = scaler.fit(data)


"""
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
"""