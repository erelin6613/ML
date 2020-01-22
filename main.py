import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #,RobustScaler,MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
#from keras.utils import to_categorical
from category_encoders import  LeaveOneOutEncoder, BinaryEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, filename='log_estimators.log')


sns.set(style="darkgrid")


def replace_nans(dataframe):

	for each in dataframe.columns:
		if each == 'id':
			continue
		dataframe.loc[:, each] = dataframe.fillna(dataframe[each].mode()[0])

	return dataframe


def encoder(dataframe, columns, enc_type='bin'):

	if enc_type == 'bin':
		for col in columns:
			unique = dataframe[col].unique()
			dataframe.loc[:, col] = \
			dataframe[col].apply(lambda x: 1 if x==unique[0] else (0 if x==unique[1] else None))
	if enc_type == 'ord':
		encoder = OrdinalEncoder(dtype=np.int16)
		for col in columns:
			dataframe.loc[:, col] = encoder.fit_transform(np.array(dataframe[col]).reshape(-1,1))


	return dataframe



def fitter(clf, X_train, X_test, y_train, y_test):

	logging.info(clf)
	print('training ', clf)
	y_train = np.array([[target] for target in y_train])
	y_test = np.array([[target] for target in y_test])
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	logging.info('accuracy:', accuracy_score(y_test, predictions))
	print('accuracy:', accuracy_score(y_test, predictions))
	print('cross_val_score:', cross_val_score(clf, X_train, y_train))
	logging.info('cross_val_score:', str(cross_val_score(clf, X_train, y_train)))
	print('features:', rank_features(clf, X_train, y_train))
	logging.info('features:', str(rank_features(clf, X_train, y_train)), '\n')

def rank_features(estimator, X_train, y_train):
	selector = RFE(estimator, 10, step=1)
	selector = selector.fit(X_train, y_train)
	return selector.ranking_


def main():

	data = pd.read_csv('train.csv')
	#print(len(data['nom_9'].unique()))
	#exit()
	data = replace_nans(data)
	print(data.describe())
	#print(data.columns)

	ord_cols = ['nom_0','nom_1', 'nom_2', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
	bin_cols = ['bin_3', 'bin_4']

	# done with: 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 
	#				'nom_1', 'nom_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'
	# not yet encoded:'nom_7', 'nom_8', 'nom_9',

	for enc in ['nom_3','nom_4', 'nom_7']:
		enc1 = pd.get_dummies(data[enc], prefix=enc)
		data.drop(columns=enc, inplace=True)
		data = pd.concat([data, enc1], axis=1)


	data = encoder(data, ord_cols, enc_type='ord')
	data = encoder(data, bin_cols, enc_type='bin')
	time_features = ['day', 'month']

	for feature in time_features:
		data[feature+'_sin'] = np.sin((2*np.pi*data[feature])/max(data[feature]))
		data[feature+'_cos'] = np.cos((2*np.pi*data[feature])/max(data[feature]))

	data.drop(time_features, axis=1, inplace=True)

	target = data['target']
	features = data.drop(['target', 'id', 'nom_5', 'nom_6', 'nom_8', 'nom_9'], axis=1).values
	print(features)
	#pca = PCA(n_components=50)
	#pca.fit(features)
	#features = pca.transform(features)
	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

	#clf_1 = SVC(verbose=True)
	clf_2 = LogisticRegression(verbose=1)
	clf_3 = GradientBoostingClassifier(verbose=1)
	clf_4 = PassiveAggressiveClassifier(verbose=1)

	#fitter(clf_1, X_train, X_test, y_train, y_test)
	fitter(clf_2, X_train, X_test, y_train, y_test)
	fitter(clf_3, X_train, X_test, y_train, y_test)
	fitter(clf_4, X_train, X_test, y_train, y_test)



main()