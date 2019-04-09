import numpy as np
from numpy import linalg as LA

def k_rank(G,k):
	wk = 0
	n,n = G.shape
	w, v = LA.eig(G)
	for i in range(k):
		wk += w[i]*np.dot(np.reshape(v[:,i],(n,1)),np.reshape((v[:,i]).T,(1,n)))
	return wk

def nystrom(G,c,k):
	n,n = G.shape
	S = np.zeros((n,c))
	D = np.zeros((c,c))

	p = [G[i,i]**2 for i in range(n)]
	p = p/np.sum(p)

	for t in range(c):
		x = np.random.multinomial(1, p, size=1)
		i_t = np.where(x==1)[1][0]
		D[t,t] = 1/np.sqrt(c*p[i_t]) 

	print(G.shape, S.shape, D.shape)
	C = np.matmul(G,S)
	C = np.matmul(C,D)
	W_temp = np.matmul(D,S.T)
	W_temp = np.matmul(W_temp,G)
	W = np.matmul(W_temp,S,D)
	W_k = k_rank(W,k)
	W_k_pinv = np.linalg.pinv(W_k)
	G_k = np.matmul(C,W_k_pinv)
	G_k = np.matmul(G_k,C.T)
	return G_k


def main():
	N = 3
	a = np.random.rand(N, N)
	G = np.tril(a) + np.tril(a, -1).T
	c = 2
	k = 2
	G_k = nystrom(G,c,k)
	print(G)
	print(G_k)

if __name__ == '__main__':
  main()