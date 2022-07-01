import numpy as np
import numdifftools as nd


class Optimization():
	"""
	Optmization class for the project:
	
	Currently optimized for 2d only.
	
	Using 8,
	"""
	
	def __init__(self, starting_point, func, runs=100, eps=1 / np.power(10, 5)):
		self.starting_point = starting_point
		self.eps = eps
		self.func = func
		self.runs = runs
	
	def estimate_deriative(self, x):
		# 3. Deriatives
		
		deriative = np.ndarray(shape=x.shape)
		for i, xn in enumerate(x):
			epsilon_vector = np.zeros_like(x)
			epsilon_vector[i] = self.eps
			deriative[i] = (self.func(x + epsilon_vector) - self.func(x - epsilon_vector)) / (2 * self.eps)
		
		return deriative
	
	def zoom(self, alpha_low, alpha_high, phi, pk, phi_0, phi_d_0, c1, c2, x):
		for a in range(10000):
			alpha_j = (alpha_low + alpha_high) / 2
			phi_j = phi(alpha_j)
			if phi_j > phi_0 + c1 * alpha_j * phi_d_0 or phi_j >= phi(alpha_low):
				alpha_high = alpha_j
			else:
				alpha_d_j = self.estimate_deriative(x + alpha_j * pk) @ pk
				if np.abs(alpha_d_j) <= -c2 * phi_d_0:
					return alpha_j
				if alpha_d_j * (alpha_high - alpha_low) >= 0:
					alpha_high = alpha_low
				alpha_low = alpha_j
		return "Done"
	
	def calc_step_length(self, x, pk, deri_x: np.ndarray, x_old=None, alpha_old=None, pk_old=None):
		# 5. Wolfe Conditions
		c1 = self.eps
		c2 = 0.8  # as written on page 162
		
		new_step = 1
		
		if alpha_old and True:  # Set False for Netwon
			new_step = alpha_old * ((self.estimate_deriative(x_old) @ pk_old) / (deri_x @ pk))  #
		alpha_old = 0
		alpha_max = 1000
		
		def phi(alpha_i):
			return self.func(x + alpha_i * pk)
		
		for alpha in np.linspace(new_step, alpha_max, 100):
			
			phi_i = self.func(x + alpha * pk)
			phi_d_i = self.estimate_deriative(x + alpha * pk) @ pk
			phi_0 = self.func(x)
			phi_d_0 = deri_x.T @ pk
			
			if phi_i > phi_0 + c1 * alpha * phi_d_i or (phi_i >= phi(alpha_old) and alpha != 0):
				return self.zoom(alpha_old, alpha, phi, pk, phi_0, phi_d_0, c1, c2, x)
			
			if np.abs(phi_d_i) <= -c2 * deri_x.T @ pk:
				return alpha
			
			if phi_d_i >= 0:
				return self.zoom(alpha, alpha_old, phi, pk, phi_0, phi_d_0, c1, c2, x)
			
			alpha_old = alpha
		
		raise EOFError('NO alpha Found')
	
	def steepest_descent(self):
		"""
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		x = self.starting_point
		
		deri_start = self.estimate_deriative(x)
		my_start_norm = np.linalg.norm(deri_start)
		x_old = None
		alpha = None
		pk_old = None
		iterations = []
		for k in range(self.runs):
			# Maximal 4 runs
			iterations.append(x)
			deri_x = self.estimate_deriative(x)
			pk = -deri_x
			alpha = self.calc_step_length(x, pk, deri_x, x_old, alpha, pk_old)
			if alpha == "Done":
				print("Stuck in Local Minima")
				print(len(iterations))
				return iterations
			
			# 1. Stopping Criteria
			if np.allclose(deri_x, 0, atol=self.eps) or np.linalg.norm(deri_x) <= self.eps * (1 + my_start_norm):
				return iterations
			
			x_old = x
			x = x + alpha * pk
			
			pk_old = pk
		print("Reached Maximal Numbers of iterations")
		print(x)
		return iterations
	
	def newton_method(self):
		"""
		:param hessian: Hessian Matrix
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		iterations = []
		x = self.starting_point
		hessian = nd.Hessian(self.func)
		
		for k in range(self.runs):
			iterations.append(x)
			deri_x = self.estimate_deriative(x)
			BK = hessian(x)
			eigen, _ = np.linalg.eig(BK)
			while np.any(eigen < 0):
				eigen, _ = np.linalg.eig(BK)
				BK = BK + np.diag(np.ones(BK.shape[0]))
			inv_hessian = np.linalg.inv(BK)
			
			pk = -(inv_hessian @ deri_x)
			
			alpha = self.calc_step_length(x, pk, deri_x)
			if alpha == "Done":
				print("Stuck in Local Minima")
				print(len(iterations))
				return iterations
			
			if np.allclose(deri_x, 0, atol=self.eps):
				return iterations
			
			x = x + alpha * pk
		# if (k % 1000 == 0 and k != 0):
		#   print(k," Iterations")
		print("Stopped after 1000 Iterations")
		return iterations
	
	def quasi_newton_method(self):
		"""
		:param hessian: Hessian Matrix
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		x = self.starting_point
		inv_hessian = np.diag(np.ones(shape=x.shape[0]))  # Taking multiply of the identity; ESTIMATE next time
		iterations = []
		for k in range(1000):
			iterations.append(x)
			deri_x = self.estimate_deriative(x)
			pk = -(inv_hessian @ deri_x)
			alpha = self.calc_step_length(x, pk, deri_x)
			
			if np.allclose(deri_x, 0, atol=self.eps):
				return iterations
			
			x = x + alpha * pk
			
			deri_xplus1 = self.estimate_deriative(x)
			sk = alpha * pk
			yk = deri_xplus1 - deri_x
			if yk.all != 0:
				rohk = 1 / (yk.T @ sk)
				
				inv_hessian = (np.diag(np.ones(shape=x.shape[0])) - (rohk * np.outer(sk, yk))) @ inv_hessian @ (
						np.diag(np.ones(shape=x.shape[0])) - rohk * np.outer(yk, sk)) + rohk * np.outer(sk,
				                                                                                        sk)  # pg 140
		
		print("run out of iterations")
		return iterations
	
	def conjugate_gradient(self):
		"""
		Multivariate Case

		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		
		# Append to list
		iterations = []
		x = self.starting_point
		deri_start = self.estimate_deriative(x)
		my_start_norm = np.linalg.norm(deri_start)
		
		for k in range(8000):
			iterations.append(x)
			grad = self.estimate_deriative(x)
			# Initalize algorithm on first run
			
			pk = -grad
			
			# [1] Jonathan R. Shewchuk, 'An introduction to the conjugate-gradient method without the agonizing pain', pp.42-43.
			# For a quick algorithm I used the simple line search. Later on I will improve this very likely.
			alpha = self.calc_step_length(x, pk, grad)
			x = x + alpha * pk
			
			gradxk1 = self.estimate_deriative(x)
			betak1 = (gradxk1 @ gradxk1) / (grad @ grad)
			pk1 = -gradxk1 + betak1 * pk
			
			if np.allclose(gradxk1, 0, atol=self.eps) or np.linalg.norm(gradxk1) <= self.eps * (1 + my_start_norm):
				iterations.append(x)
				return iterations
		print(k)
		print("Got eliminated after:", k)
		return iterations


class ResidualOptimization():
	
	def __init__(self, starting_point, func, runs=300, eps=1 / np.power(10, 5)):
		self.starting_point = starting_point
		self.eps = eps
		self.func = func
		self.runs = runs
	
	def estimate_deriative(self, rjs, x):
		# nd case
		jacobian = self.calc_jacobian(rjs, x)
		
		return jacobian.T @ rjs(x)
	
	def calc_jacobian(self, rjs, x):
		m = len(rjs(x))
		jacobian = np.ndarray(shape=(m, x.shape[0]))
		
		for i, xn in enumerate(x):
			epsilon_vector = np.zeros_like(x)
			epsilon_vector[i] = self.eps
			
			jacobian[:, i] = (rjs(x + epsilon_vector) - rjs(x - epsilon_vector)) / (2 * self.eps)
		return jacobian
	
	def estimate_hessian(self, rjs, x):
		# nd case
		jacobian = self.calc_jacobian(rjs, x)
		
		return jacobian.T @ jacobian
	
	def calc_step_length(self, rjs, B, x, pk):
		jacobian = self.calc_jacobian(rjs, x)
		H = self.estimate_hessian(rjs, x)
		
		alpha = ((pk.T @ jacobian.T) @ B - pk.T @ H @ x - x.T @ H @ pk + B.T @ jacobian @ pk) / (2 * pk.T @ H @ pk)
		return alpha
	
	def steepest_descent(self, rjs, B):
		"""
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		
		x = self.starting_point
		iterations = []
		iterations.append(x)
		iterations = []
		for k in range(self.runs):
			# Maximal 4 runs
			iterations.append(x)
			deri_x = self.estimate_deriative(rjs, x)
			pk = -deri_x
			alpha = self.calc_step_length(rjs, B, x, pk)
			if alpha == "Done":
				print("Stuck in Local Minima")
				print(len(iterations))
				return iterations
			
			# 1. Stopping Criteria
			if np.allclose(self.func(x), 0, atol=1 / np.power(10, 2)):  # Using a different rule here.
				return iterations
			
			x = x + alpha * pk
			
			if (k % 10 == 0 and k != 0):
				print(k, "iterations")
		print("Reached Maximal Numbers of iterations")
		print(x)
		return (iterations)
	
	def newton_method(self, rjs, B):
		"""
		:param hessian: Hessian Matrix
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		
		x = self.starting_point
		
		iterations = []
		for k in range(self.runs):
			
			iterations.append(x)
			hessian = self.estimate_hessian(rjs, x)
			deri_x = self.estimate_deriative(rjs, x)
			inv_hessian = np.linalg.inv(hessian)
			
			pk = -(inv_hessian @ deri_x)
			
			alpha = self.calc_step_length(rjs, B, x, pk)
			
			if np.allclose(self.func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
				return iterations
			
			x = x + alpha * pk
			
			if (k % 10 == 0 and k != 0):
				print(k, "iterations")
		print("Reached Maximal Numbers of iterations")
		print(x)
		return (iterations)
	
	def conjugate_gradient(self, rjs, B):
		"""
		Multivariate Case
	
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		
		# Append to list
		x = self.starting_point
		iterations = []
		
		grad = self.estimate_deriative(rjs, x)
		pk = -grad
		for k in range(self.runs):
			iterations.append(x)
			
			alpha = self.calc_step_length(rjs, B, x, pk)
			x = x + alpha * pk
			
			gradxk1 = self.estimate_deriative(rjs, x)
			
			betak1 = (gradxk1 @ gradxk1) / (grad @ grad)
			pk = - gradxk1 + betak1 * pk
			grad = gradxk1
			
			if np.allclose(self.func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
				iterations.append(x)
				return iterations
			
			if (k % 10 == 0 and k != 0):
				print(k, "iterations")
		
		print("Reached Maximal Numbers of iterations")
		return (iterations)
	
	def quasi_newton_method(self, rjs, B):
		"""
		:param inv_hessian:
		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
		
		x = self.starting_point
		inv_hessian = np.linalg.inv(
			self.estimate_hessian(rjs, x))  # Taking multiply of the identity; ESTIMATE next time
		iterations = []
		
		for k in range(self.runs):
			
			iterations.append(x)
			deri_x = self.estimate_deriative(rjs, x)
			pk = -(inv_hessian @ deri_x)
			alpha = self.calc_step_length(rjs, B, x, pk)
			
			if np.allclose(self.func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
				return iterations
			
			x = x + alpha * pk
			
			deri_xplus1 = self.estimate_deriative(rjs, x)
			sk = alpha * pk
			yk = deri_xplus1 - deri_x
			if yk.all != 0:
				rohk = 1 / (yk.T @ sk)
				
				inv_hessian = (np.diag(np.ones(shape=x.shape[0])) - (rohk * np.outer(sk, yk))) @ inv_hessian @ (
						np.diag(np.ones(shape=x.shape[0])) - rohk * np.outer(yk, sk)) + rohk * np.outer(sk, sk)  # pg 140
			
			if (k % 10 == 0 and k != 0):
				print(k, "iterations")
		print("Reached Maximal Numbers of iterations")
		print(x)
		return (iterations)
