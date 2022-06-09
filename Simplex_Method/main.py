import numpy as np

# Enter the Matrix A
A = np.asarray([[1, 1, 1, 0], [2, 0.5, 0, 1]])

# Calculate Problem size:
n = A.shape[0]
# Choose Initial B
B_i = [2, 3]
# N = inverse of B

# Choose b as boundary
b = np.asarray([5, 8])
c = np.asarray([-2, -4, 0, 0])


# function to optimize:
def f(x):
	return -5 * x[0] - x[1]


def simplex_method(B_i, A, b, c):
	x_N = 0
	x = np.zeros(shape=(4))
	N_i = [x for x in range(0, 4) if x not in B_i]
	while True:
		
		B = A[:, B_i]
		B_inverse = np.linalg.inv(B)
		x_B = B_inverse @ b
		
		lambda_ = B_inverse.T @ c[B_i]
		s_N = c[N_i] - A[:, N_i].T @ lambda_
		print("s_N: ", s_N)
		print("lambda_: ",lambda_)
		if np.all(s_N >= 0):
			return x
		
		q = 0
		for i in range(s_N.shape[0]):
			if (s_N[i] < 0):
				q = i
				break
		d = B_inverse @ A[:, q]
		
		if np.all(d <= 0):
			print("Problem is unbounded")
			exit(0)
		
		tmp = np.Inf
		p = 0
		for i in range(d.shape[0]):
			if(d[i] == 0):
				continue
			if tmp > x_B[i] / d[i]:
				p = i
				tmp = x_B[i] / d[i]
				x_qplus = tmp
		
		q = N_i[q]
		x[q] = x_qplus
		x_Bplus = x_B - d * x_qplus
		
		x_nplus = np.zeros(shape=A.shape[0])
		x_nplus[p] = x_qplus
		
		k = B_i[p]
		a = B_i.index(B_i[p])
		B_i.remove(B_i[p])
		B_i.insert(a,q)
		a = N_i.index(q)
		N_i.remove(q)
		N_i.insert(q,k)
		
		print("x: ", x)
		print("B_i: ", B_i)
		print("N_i: ", N_i)
		
		print("NEW RUN -----")


x = simplex_method(B_i,  A, b, c)
print("Solution x: ",x)