"""
Author: Viktor Loreth
Date: 27.07.2022
Content: Simplex Method, Nocedal

Programming subject: Solving homework assignment Numerical Optimization at Kepler University
"""

import numpy as np
import scipy.linalg as la


class Simplex:
	def __init__(self, A, b, c):
		# Concatenate slack variables
		self.s_N = None
		A = np.concatenate((A, np.diag(v=np.ones(shape=A.shape[0]))), axis=1)
		self.A = A
		self.init = True
		# Choose Initial B = slack variables
		self.B_i = list(range(A.shape[1] - A.shape[0], A.shape[1]))
		self.b = b
		c = np.concatenate((c, np.zeros(shape=A.shape[1] - c.shape[0])))
		self.c = c
		self.x = np.zeros_like(c)
		self.size = A.shape[1]
	
	def function_value(self, x):
		return self.c @ self.x
	
	def feasible_start(self):
		# One could improve this method by deleting slack variables with E[i,i] = 1
		# This is not done due it costing time and it barely improves the time efficiency.
		
		E = np.zeros(shape=(b.shape[0], b.shape[0]))
		z = np.copy(self.b)
		for i, b_ele in enumerate(self.b):
			if b_ele >= 0:
				E[i, i] = 1
			else:
				E[i, i] = -1
				z[i] = - b_ele
		
		# x starting point is equal to c
		self.x = np.zeros_like(self.c)
		self.x = np.concatenate((self.x, z))
		
		# concatenate slack variables
		self.B_i = list(range(self.A.shape[1], self.A.shape[1] + E.shape[1]))
		self.A = np.concatenate((self.A, E), axis=1)
		self.c = np.concatenate(((self.c), np.zeros_like(z)))
		x = self.simplex_run()
	
	def simplex_run(self):
		print("A=\n", self.A[:, :c.shape[0]])
		print("b=\n", self.b)
		print("c=\n", self.c[:self.size])
		N_i = [_ for _ in range(0, self.c.shape[0]) if _ not in self.B_i]  # All indices not in B are in N
		self.lu_update()
		while True:
			# Given step
			
			# Solve for Lambda and s_N
			self.x[N_i] = 0
			self.lambda_ = self.solve_lu_system(self.c[self.B_i], transpose=True)
			self.s_N = self.c[N_i] - self.A[:, N_i].T @ self.lambda_
			# check if all function values are positive
			if np.all(self.s_N >= 0):
				return self.x
			
			# find indices which are negative
			
			for i in range(self.s_N.shape[0]):
				if (self.s_N[i] < 0):
					self.q = N_i[i]  # Entering indice
					break
			
			# Find values which solve the equation
			
			d = self.solve_lu_system(self.A[:, self.q])
			
			if np.all(d < 0):
				print("Problem is unbounded")
				exit(0)
			
			# find maximum step size to minimize
			tmp = np.Inf
			cycling = True
			for i in range(d.shape[0]):
				
				if d[i] <= 0:
					if cycling:
						self.p = self.B_i[i]
						continue
				elif tmp > self.x[self.B_i][i] / d[i]:
					self.p = self.B_i[i]  # p = Minimizing column
					tmp = self.x[self.B_i][i] / d[i]
					x_qplus = tmp
					cycling = False
			
			# Update LU to solve the linear system
			
			# Adjust x to new values
			self.x[self.B_i] = self.x[self.B_i] - d * x_qplus
			self.x[self.q] = x_qplus
			
			# Adjust basis
			
			a = self.B_i.index(self.p)
			
			self.B_i.remove(self.p)
			self.B_i.insert(a, self.q)
			
			a = N_i.index(self.q)
			N_i.remove(self.q)
			
			if self.p > self.size:
				self.A[:, self.p] = 0
			else:
				N_i.insert(a, self.p)
			
			self.lu_update()
			self.stat()
	
	# self.stat()
	
	# This part of code is used to save efficiency by using a LU factorization.
	def lu_update(self):
		if True or self.init:  # Normally one should only use self.init, but in our case I am not using the permutation.
			self.P, self.L, self.U = la.lu(self.A[:, self.B_i], check_finite=False)
			self.init = False
		else:
			# Generate Permutation Matrix
			PMatrix = np.diag(np.ones_like(self.L[0]))
			tmp = PMatrix[self.q].copy()
			PMatrix[self.q:-1] = PMatrix[self.q + 1:]
			PMatrix[-1] = tmp
			PMatrix = PMatrix.T  # Because we are using rows instead of columns
			
			L_inv = np.linalg.inv(self.L)
			w_row = L_inv @ self.A[:, self.q]
			
			# Permutation
			Z = PMatrix @ L_inv @ self.A[:, self.B_i] @ PMatrix.T
			
			# Reconstructing factors
			P, L1, U1 = la.lu(Z, check_finite=False)
			
			self.L = self.L @ PMatrix.T @ L1
			self.U = U1 @ PMatrix
	
	def solve_lu_system(self, b_, transpose=False):
		if transpose:
			self.L = self.L.T
			self.U = self.U.T
			x = self.back_sub(self.forward_sub(b_))
			self.L = self.L.T
			self.U = self.U.T
		else:
			x = self.back_sub(self.forward_sub(b_))
		return x
	
	def forward_sub(self, b_):
		"""x = forward_sub(L, b) is the solution to L x = b
			L must be a lower-triangular matrix
			b must be a vector of the same leading dimension as L
	
			Took it from: https://courses.engr.illinois.edu/cs357/fa2019/references/ref-7-linsys/
			as it is trivial.
		"""
		n = self.L.shape[0]
		x = np.zeros(n)
		for i in range(n):
			tmp = b_[i]
			for j in range(i):
				tmp -= self.L[i, j] * x[j]
			x[i] = tmp / self.L[i, i]
		return x
	
	def back_sub(self, b_):
		"""x = back_sub(U, b) is the solution to U x = b
			U must be an upper-triangular matrix
			b must be a vector of the same leading dimension as U
		"""
		n = self.U.shape[0]
		x = np.zeros(n)
		for i in range(n - 1, -1, -1):
			tmp = b_[i]
			for j in range(i + 1, n):
				tmp -= self.U[i, j] * x[j]
				
			if self.U[i,i] == 0:
				print("Division by 0 occured, aborting")
				exit(1)
			x[i] = tmp / self.U[i, i]
		return x
	
	def stat(self):
		print(f"Current x = {self.x[:self.size]}")
		print(f"Current g = {self.x[self.size:]}")
		print(f"Function value = {self.function_value(self.x)}")
		print(f"S_N = {self.s_N}")
	# print(f"B_i = {self.B_i}")


# Enter the Matrix A
A = np.asarray([[1, 0, 0, 1, 0, 1, 2, 2, 1, 1],
                [0, -1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                [2, 0, 0, 1, 0, 0, 1, 0, 0, 2],
                [1, 0, 0, 0, 1, 2, 2, 0.5, 0, 1]])
# Define Boundary
b = np.asarray([6, -4,3, 2, 5])
# define function value
c = np.asarray([-2, 1, 1, -2, -1, -2, -4, 2, -4, -2])

my_simplex = Simplex(A, b, c)

# Finds a feasible starting point automatically
#my_simplex.feasible_start()
# Runs a method with starting point x = 0
solution = my_simplex.simplex_run()

my_simplex.stat()
