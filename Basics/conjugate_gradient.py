# Implements different algorithms for conjugate_gradient
import numpy as np
import matplotlib.pyplot as plt


# CG preliminary version

def cg_preliminary_version(A, b, x_k):
	r_k = A @ x_k - b
	k = 0
	p_k = -r_k
	while (np.sum(np.abs(r_k)) > 10 ** -6):
		alpha_k = (r_k.T @ r_k) / (p_k.T @ A @ p_k)
		x_k1 = x_k + alpha_k * p_k
		r_k1 = r_k + alpha_k * A @ p_k
		beta_k1 = (r_k1.T @ r_k1) / (r_k.T @ r_k)
		p_k1 = -r_k1 + beta_k1 * p_k
		
		k += 1
		x_k = x_k1
		beta_k = beta_k1
		p_k = p_k1
		r_k = r_k1
		descend.append(x_k)
	print(k)


def cg_version(A, b, x_k):
	r_k = A @ x_k - b
	k = 0
	p_k = -r_k
	while (np.abs(r_k).sum() > 10 ** -6):
		alpha_k = (r_k.T @ r_k) / (p_k.T @ A @ p_k)
		x_k1 = x_k + alpha_k * p_k
		r_k1 = r_k + alpha_k * A @ p_k
		beta_k1 = (np.dot(r_k1, r_k1)) / (np.dot(r_k, r_k))
		p_k1 = -r_k1 + beta_k1 * p_k
		
		k += 1
		x_k = x_k1
		beta_k = beta_k1
		p_k = p_k1
		r_k = r_k1
		descend.append(x_k)
		###Improving!
		print('Residual Error: ',np.abs(r_k).sum())
		if k == A.shape[0]:
			r_k = np.round(r_k1, 2)
	print('iterations', k)


def hilbert(n):
	x = np.arange(1, n + 1) + np.arange(0, n)[:, np.newaxis]
	return 1.0 / x


def unclustered_dia(n):
	# Returns a nxn matrix without clusters
	
	return np.diag(np.arange(1, n + 1))


# A = unclustered_dia(k)


# print("Matrix", A)
# x_k = np.zeros(shape=A.shape[0])
# descend = [x_k]
# b = np.ones(shape=A.shape[0]).T


# cg_version(A, b, x_k)


def clustered_dia(n, k):
	#n dimension, k clusters
	clusters = int(n / k)
	my_vector = np.array([])
	for a in range( clusters):
		a= a*a
		my_vector = np.append(my_vector, np.linspace(a + 1, a + 1.1, clusters))
	
	return np.diag(my_vector)


A = clustered_dia(30, 10)


x_k = np.zeros(shape=A.shape[0])
descend = [x_k]
b = np.ones(shape=A.shape[0]).T
print('Test with 60x60 and 6 clusters')
cg_version(A, b, x_k)
