from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from nystrom import *

def builtinNystroem(G, c, g):
	feature_map_subset = Nystroem(gamma = g, n_components = c)
	gram_approx = feature_map_subset.fit_transform(G)
	return np.matmul(gram_approx, gram_approx.T)

def checker(k, c, g, rank):
	mat1 = builtinNystroem(k, c, g)
	# print(k.shape, k.dtype, g)
	kernel = rbf_kernel(k, gamma = g)
	mat2,_ = nystrom(kernel, c, rank)
	metric1 = np.linalg.norm(kernel - mat1, ord = 'fro')/np.linalg.norm(kernel)
	metric2 = np.linalg.norm(kernel - mat2, ord = 'fro')/np.linalg.norm(kernel)
	return metric1, metric2
	
def main():
	iris = datasets.load_iris()
	mat = iris.data
	# c = 55
	datapoints = np.arange(50,151)
	rank = 42
	# ranks = np.arange(1,55)
	gamma = .25
	# values = np.arange(1,1000)/100
	met1 = []
	met2 = []
	# for g in values:
	# 	temp1, temp2 = checker(mat, c, g, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)
	# plt.plot(values, met1, c='black', label = 'Built-in Nystroem')
	# plt.plot(values, met2, c='red', label = 'Our implementation')
	# plt.xlabel('Gamma values')
	# plt.ylabel('Error Metric')
	# plt.legend()
	# plt.title('Sklearn implementation vs our implementation \n(c = 55, rank = 42)')
	# plt.show()

	# ind, mini = 0, float('inf')

	for c in datapoints:
		temp1, temp2 = checker(mat, c, gamma, rank)
		met1.append(temp1)
		met2.append(temp2)
		# if mini > temp2 - temp1:
		# 	mini = temp2-temp1
		# 	ind = c

	# print(ind)
	plt.plot(datapoints, met1, c='black', label = 'Built-in Nystroem')
	plt.plot(datapoints, met2, c='red', label = 'Our implementation')
	plt.xlabel('Number of Data points')
	plt.ylabel('Error Metric')
	plt.legend()
	plt.title('Sklearn implementation vs our implementation \n(gamma = .25, rank = 42)')
	plt.show()

	# for rank in ranks:
	# 	temp1, temp2 = checker(mat, c, gamma, rank)
	# 	met1.append(temp1)
	# 	met2.append(temp2)
	# # 	if mini > temp2 - temp1:
	# # 		mini = temp2 - temp1
	# # 		ind = rank

	# # print(ind)

	# plt.plot(ranks, met1, c='black', label = 'Built-in Nystroem')
	# plt.plot(ranks, met2, c='red', label = 'Our implementation')
	# plt.xlabel('Rank of approximations')
	# plt.ylabel('Error Metric')
	# plt.legend()
	# plt.title('Sklearn implementation vs our implementation \n(c = 55, gamma = 0.25)')
	# plt.show()

if __name__ == '__main__':
	main()