import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import math
import matplotlib.pyplot as plt


def distance(a , b, disctance_type = 'euclidian'):
	return abs(np.linalg.norm(a-b))

def kmeans(dataset = np.ndarray, k = 5, epsilon = 0, disctance_type = 'euclidian'):
	data_instances = dataset.shape[0]
	data_features = dataset.shape[1]
	prototypes = dataset[np.random.randint(0, data_instances - 1, size=k)]
	prototypes_old = np.zeros(prototypes.shape)
	belongs_to = np.zeros((data_instances, 1))
	if disctance_type == 'euclidian':
		norm = distance(prototypes, prototypes_old)
	while norm > epsilon:
		norm = distance(prototypes, prototypes_old)
		prototypes_old = prototypes
		for index_instance, instance in enumerate(dataset):
			dist_vec = np.zeros((k, 1))
			for index_prototype, prototype in enumerate(prototypes):
				dist_vec[index_prototype] = distance(prototype, instance)

			belongs_to[index_instance, 0] = np.argmin(dist_vec)

		tmp_prototypes = np.zeros((k, data_features))

		for index in range(len(prototypes)):
			instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
			prototype = np.mean(dataset[instances_close], axis=0)
			tmp_prototypes[index, :] = prototype

		prototypes = tmp_prototypes

	return prototypes, belongs_to, k

def kmeans_predict(dataset, test_features, prototypes=None, belongs_to=None):
	prototypes, belongs_to, k = kmeans(dataset, k = 3)
	print(prototypes)
	print('************')
	print(belongs_to)
	for i in range(1, k):
		if distance(test_features, prototypes[i]) < distance(test_features, prototypes[i-1]):
			prediction = belongs_to[i, :]
		else:
			pass

	return prediction


def Main():

	X, y = make_blobs(n_samples=170, centers=3, n_features=2, random_state=30)
	print(type(X))
	test_features = [2, -8]
	plt.scatter(test_features[0], test_features[1], c = 'black', s = 150)
	for i in range(len(X)):
		if y[i] == 0:
			plt.scatter(X[i][0], X[i][1], c = 'b')
		if y[i] == 1:
			plt.scatter(X[i][0], X[i][1], c = 'r')
		if y[i] == 2:
			plt.scatter(X[i][0], X[i][1], c = 'y')
	plt.show()
	prototypes, belongs_to, k = kmeans(X, k = 3)
	prediction = kmeans_predict(dataset = X, test_features = test_features, prototypes = prototypes, belongs_to = belongs_to)
	print(prediction)


if __name__ == '__main__':
	Main()
