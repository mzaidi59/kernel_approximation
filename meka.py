import numpy as np
from numpy import array
from numpy import zeros
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot as plt
from nystrom import *



def meka(X, n_c, g):
	G_k = np.zeros((X.shape[0],X.shape[0]))
	k = int(0.3 * X.shape[0] /n_c)
	kmeans = KMeans(n_clusters=n_c, random_state=0).fit(X)
	thresh = 1.2
	#Cluster identity of each row
	n_clus = kmeans.predict(X)
	# print(n_clus)
	X_p = np.zeros_like(X)
	in_p = 0
	n_p = np.zeros(n_c)
	s_in = np.zeros(n_c, dtype=int)
	e_in = np.zeros(n_c, dtype=int)
	W = []
	X_cen = np.zeros((n_c,X.shape[1]))

	for i in range(n_c):
		ind = np.argwhere(n_clus == i)
		n_p[i] = int(ind.shape[0])
		for j in ind:
			X_p[in_p,:] = X[j[0],:]
			in_p += 1
			X_cen[i,:] += X[j[0],:]
		X_cen[i,:] = X_cen[i]/n_p[i]
		s_in[i] = np.sum(n_p[:i])
		e_in[i] = s_in[i] + n_p[i]
		G_temp = rbf_kernel(X_p[int(s_in[i]):int(e_in[i]),:], gamma = g)
		G_k[s_in[i]:e_in[i], s_in[i]:e_in[i]], W_t = nystrom(G_temp, int(np.ceil(n_p[i]*0.4)) ,int(np.ceil(n_p[i]*0.9)))
		W.append(W_t)
	
	G = rbf_kernel(X_p, gamma = g)
	for i in range(n_c):
		for j in range(n_c):
			print(np.linalg.norm(X_cen[i]-X_cen[j]))
			if not (i==j) and np.linalg.norm(X_cen[i]-X_cen[j]) < thresh and not (i==j):

				W_a = W[i]
				W_b = W[j]
				G_hat = G[s_in[i]:e_in[i],s_in[j]:e_in[j]]

				#------------------------------------
				M_1 = np.linalg.pinv(W_a.T.dot(W_a))
				M_2 = M_1.dot(W_a.T)
				M_3 = M_2.dot(G_hat)
				M_l = np.linalg.pinv(W_b.T.dot(W_b))
				M_4 = M_3.dot(W_b)
				M_5 = M_4.dot(M_l)
				M_6 = W[i].dot(M_5)
				M_f = M_6.dot(W[j].T)
				#------------------------------------

				#--------------------------------------------
				G_k[s_in[i]:e_in[i], s_in[j]:e_in[j]] = M_f
				G_k[s_in[j]:e_in[j], s_in[i]:e_in[i]] = M_f.T
				#--------------------------------------------
	print(G.shape, G_k.shape)
	return G, G_k

# def main():
# 	gamma = 0.25
# 	x = range(100, 201)
# 	met1 = []
# 	met2 = []
# 	for i in x:
# 		low, high = i/4, i/2
# 		m, n = i, np.random.randint(low, high)
# 		n_c = 5
# 		X = np.random.random((m,n))
# 		k, k_app1 = meka(X, n_c, gamma)
# 		metric1 = np.linalg.norm(k-k_app1, ord = 'fro')/np.linalg.norm(k, ord = 'fro')

# 		c = np.random.randint(90, i-1)
# 		rank = np.random.randint(90, i)
# 		k_app2,_ = nystrom(k, c, rank)
# 		metric2 = np.linalg.norm(k-k_app2, ord = 'fro')/np.linalg.norm(k, ord = 'fro')

# 		met1.append(metric1)
# 		met2.append(metric2)

# 	plt.scatter(x, met1, c = 'black')
# 	plt.xlabel('Rows of input matrix(X)')
# 	plt.ylabel('Difference between Metrics')
# 	plt.show()

# if __name__ == '__main__':
# 	main()
