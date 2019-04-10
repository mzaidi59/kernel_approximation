import numpy as np
from numpy import linalg as LA
from sklearn.datasets import make_spd_matrix
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
#

def k_rank(G,k):
	# wk = 0
	# n,n = G.shape
	# w, v = LA.eig(G)
	# print(w)
	# for i in range(k):
	# 	wk += w[i]*np.dot(np.reshape(v[:,i],(n,1)),np.reshape((v[:,i]).T,(1,n)))
	# return wk
	U, s, VT = svd(G)
	s[k:] = 0
	# create m x n Sigma matrix
	Sigma = zeros((G.shape[0], G.shape[1]))
	# populate Sigma with n x n diagonal matrix
	Sigma[:G.shape[1], :G.shape[1]] = diag(s)
	# reconstruct matrix
	B = U.dot(Sigma.dot(VT))
	
	return B

def nystrom(G,c,k):
	n,_ = G.shape
	S = np.zeros((n,c))
	D = np.zeros((c,c))

	p = [G[i,i]**2 for i in range(n)]
	p = p/np.sum(p)

	for t in range(c):
		x = np.random.multinomial(1, p, size=1)
		i_t = np.where(x==1)[1][0]
		if p[i_t] < 0:
			print(p[i_t])
		D[t,t] = 1/np.sqrt(c*p[i_t]) 
		S[i_t,t] =1


	C = np.matmul(G,S)
	C = np.matmul(C,D)
	W_temp = np.matmul(D,S.T)
	W_temp = np.matmul(W_temp,G)
	W = np.matmul(W_temp,S,D)
	print("----------")
	print(C)
	print("----------")
	print(W)
	W_k = k_rank(W,k)
	print("----------")
	print(W_k)
	W_k_pinv = np.linalg.pinv(W_k)
	G_k = np.matmul(C,W_k_pinv)
	G_k = np.matmul(G_k,C.T)
	return G_k


def main():
	N = 100
	# a = np.random.rand(N, N)
	# G = np.tril(a) + np.tril(a, -1).T

	G = make_spd_matrix(10)
	c = 9
	k = 4
	G_k = nystrom(G,c,k)
	print("----------")
	print(G)
	print("----------")	
	print(G_k)

if __name__ == '__main__':
  main()