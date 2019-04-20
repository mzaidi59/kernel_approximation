import numpy as np
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
#


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def k_rank(W,k):

	U, s, VT = svd(W)
	# print(U-VT.T)
	s[k:] = 0
	# create m x n Sigma matrix
	Sigma = np.zeros_like(W)
	# populate Sigma with n x n diagonal matrix
	Sigma[:W.shape[1], :W.shape[1]] = diag(s)
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


	C_1 = G.dot(S)
	C = C_1.dot(D)

	W_1 = D.dot(S.T)
	W_2 = W_1.dot(G)
	W_3 = W_2.dot(S)
	W = W_3.dot(D)

	# print("----------")
	# print(C)
	# print("----------")
	# print(W)
	# print(check_symmetric(W))
	W_k = k_rank(W,k)
	# print(np.linalg.inv(C.dot(C.T)))
	# print("----------")
	# print(W_k)
	W_k_pinv = np.linalg.pinv(W_k)
	G_k = np.matmul(C,W_k_pinv)
	G_k = np.matmul(G_k,C.T)
	return G_k,C


# def main():
# 	N = 1000
# 	# a = np.random.rand(N, N)
# 	# G = np.tril(a) + np.tril(a, -1).T

# 	G = make_spd_matrix(N)+10e-5*np.identity(N)
# 	c = int(0.5 * N)
# 	k = int(0.1 * N)
# 	G_k,C = nystrom(G,c,k)
# 	# print("----------"prin
# 	# print(G)
# 	# print("---------")
# 	# print(G_k)
# 	print(np.linalg.norm(G-G_k)/np.linalg.norm(G))

# if __name__ == '__main__':
#   main()