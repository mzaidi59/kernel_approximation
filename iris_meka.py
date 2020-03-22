import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from nystrom import *
from meka import *
import time

n_time = 0
m_time = 0

def checker(mat, n_clus, gamma, thresh, n_points, rank):
	global m_time
	global n_time

	start = time.time()
	kernel, mat_meka = meka(mat, n_clus, gamma, thresh)
	end = time.time()
	m_time += end - start

	start = time.time()
	mat_nystroem, _ = nystrom(kernel, n_points, rank)
	end = time.time()
	n_time += end - start

	metric1 = np.linalg.norm(kernel - mat_nystroem, ord = 'fro')/np.linalg.norm(kernel, ord = 'fro')
	metric2 = np.linalg.norm(kernel - mat_meka, ord = 'fro')/np.linalg.norm(kernel, ord = 'fro')
	return metric1, metric2


def main():
	iris = datasets.load_iris()
	mat = iris.data
	gamma = 6
	# gamma = np.arange(1, 201)/2
	c = 100
	# datapoints = np.arange(51, 151)
	rank = 47
	# ranks = np.arange(1, 100)
	cluster = 5
	# clusters = range(1,101)
	# thresh = 9
	threshs = np.arange(10, 100)/10
	met1 = []
	met2 = []
	minima = float('inf')
	ind = 0
	# for n in clusters:
	# 	temp1, temp2 = checker(mat, n, gamma, thresh, c, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)

	# plt.scatter(clusters, met2, c = 'red', label = 'MEKA metric')
	# plt.scatter(clusters, met1, c = 'black', label = 'Nystroem metric')
	# plt.xlabel('Number of clusters')
	# plt.ylabel('Error Metric')
	# plt.title('Nystroem vs MEKA \n(datapoints = 100, rank = 47, gamma = 6, threshhold = 9)')
	# plt.legend()
	# plt.show()

	# for g in gamma:
	# 	temp1, temp2 = checker(mat, cluster, g, thresh, c, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)
	# # print("Plotting")
	# plt.scatter(gamma, met2, c = 'red', label = 'MEKA metric')
	# plt.scatter(gamma, met1, c = 'black', label = 'Nystroem metric')
	# plt.xlabel('Value of Gamma')
	# plt.ylabel('Error Metric')
	# plt.title('Nystroem vs MEKA \n (clusters = 5, rank = 47, datapoints = 100, threshhold = 9)')
	# plt.legend()
	# plt.show()

	# for rank in ranks:
	# 	temp1, temp2 = checker(mat, cluster, gamma, thresh, c, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)

	# print('Time taken for Nystroem = ', n_time, 'secs')
	# print('Time taken for Meka = ', m_time, 'secs')

	# plt.scatter(ranks, met2, c = 'red', label = 'MEKA metric')
	# plt.scatter(ranks, met1, c = 'black', label = 'Nystroem metric')
	# plt.xlabel('Rank used for approximation')
	# plt.ylabel('Error Metric')
	# plt.title('MEKA vs Nystroem\n(datapoints = 100, cluster = 5, gamma = 6, threshhold = 9)')
	# plt.legend()
	# plt.show()


	# for c in datapoints:
	# 	temp1, temp2 = checker(mat, cluster, gamma, thresh, c, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)

	# 	if minima > temp2:
	# 		minima = temp2
	# 		ind = c

	# print('*****', minima, ind, '*****')
	# plt.scatter(datapoints, met2, c = 'red', label = 'MEKA metric')
	# plt.scatter(datapoints, met1, c = 'black', label = 'Nystroem metric')
	# plt.xlabel('Number of datapoints used for Nystroem')
	# plt.ylabel('Error Metric')
	# plt.title('Nystroem vs MEKA \n(gamma = 6, rank = 47, clusters = 5, threshhold = 9)')
	# plt.legend()
	# plt.show()

	for thresh in threshs:
		temp1, temp2 = checker(mat, cluster, gamma, thresh, c, rank)
		met1.append(temp1)
		met2.append(temp2)
	print("Plotting")
	plt.scatter(threshs, met2, c = 'red', label = 'MEKA metric')
	plt.scatter(threshs, met1, c = 'black', label = 'Nystroem metric')
	plt.xlabel('Value of threshhold (MEKA)')
	plt.ylabel('Error Metric')
	plt.title('Nystroem vs MEKA \n (gamma = 6, clusters = 5, rank = 41, datapoints = 50)')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	main()