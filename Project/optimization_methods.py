import numpy as np
import numdifftools as nd
import sys

eps = 1 / np.power(10, 5)

sys.setrecursionlimit(3000)


def estimate_deriative(func, x, epsilon=eps):
	deriative = np.ndarray(shape=x.shape)
	# nd case
	if x.size != 1:
		print(x)
		for i, xn in enumerate(x):
			epsilon_vector = np.zeros_like(x)
			epsilon_vector[i] = epsilon
			
			deriative[i] = (func(x + epsilon_vector) - func(x - epsilon_vector)) / (2 * epsilon)
	# 1d case
	else:
		
		deriative = np.array((func(x + epsilon) - func(x - epsilon)) / (2 * epsilon))
	
	return deriative


def estimate_hessian_m_vector(func, x, vector, epsilon=eps):
	hessian_p = (estimate_deriative(func, x, epsilon * vector) - estimate_deriative(func, x)) / epsilon
	return hessian_p


def calc_step_length(func, x, pk, deri_x: np.ndarray, condition='Wolfe'):
	if condition == 'Wolfe':
		c1 = eps
		c2 = 0.6  # as written on page 162
		# nd case
		if x.size != 1:
			for alpha in np.linspace(c1, 1, 10000):
				if func(x + alpha * pk) <= func(x) + c1 * alpha * deri_x.T @ pk and \
						estimate_deriative(func, x + alpha * pk).T @ pk >= c2 * estimate_deriative(func, x).T @ pk:
					return alpha
			raise EOFError('NO alpha Found')
		# 1d CASE
		else:
			for alpha in np.linspace(c1, 1, 10000):
				if func(x + alpha * pk) <= func(x) + c1 * alpha * deri_x * pk and \
						estimate_deriative(func, x + alpha * pk) * pk >= c2 * estimate_deriative(func, x) * pk:
					return alpha
			raise EOFError('NO alpha Found')
	
	elif condition == 'Goldstein':
		raise EnvironmentError("Not Implemented")


def steepest_descent_1d(func, x, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	iterations.append(float(x))
	deri_x = estimate_deriative(func, x, epsilon)
	pk = -deri_x
	alpha = calc_step_length(func, x, pk, deri_x)
	
	if np.allclose(pk, 0, stopping_criteria):
		return iterations
	
	else:
		return steepest_descent_1d(func, x + alpha * pk, iterations, epsilon, stopping_criteria)


def steepest_descent(func, x, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	
	iterations.append(x)
	deri_x = estimate_deriative(func, x, epsilon)
	pk = -deri_x
	alpha = calc_step_length(func, x, pk, deri_x)
	if np.allclose(deri_x, 0, atol=stopping_criteria):
		return iterations
	
	else:
		return steepest_descent(func, x + alpha * pk, iterations, epsilon, stopping_criteria)


def newton_method_1d(func, x, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param hessian: Hessian Matrix
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	
	hessian = nd.Hessian(func)
	hessian = hessian(x)
	iterations.append(float(x))
	deri_x = estimate_deriative(func, x, epsilon)
	
	if x.size == 1:
		hessian = hessian.flatten()
		inv_hessian = 1 / hessian
		pk = -(inv_hessian * deri_x)
	else:
		inv_hessian = np.linalg.inv(hessian)
		pk = -(inv_hessian @ deri_x)
	
	alpha = calc_step_length(func, x, pk, deri_x)
	
	if np.allclose(pk, 0, stopping_criteria):
		return iterations
	
	else:
		return newton_method_1d(func, x + alpha * pk, iterations, epsilon, stopping_criteria)


def newton_method(func, x, hessian=None, iterations=None, epsilon=eps, stopping_criteria=eps):
	"""
	:param hessian: Hessian Matrix
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	if iterations is None:
		iterations = []
	if hessian is None:
		hessian = nd.Hessian(func)
	
	iterations.append(x)
	deri_x = estimate_deriative(func, x, epsilon)
	inv_hessian = np.linalg.inv(hessian(x))
	
	pk = -(inv_hessian @ deri_x)
	
	alpha = calc_step_length(func, x, pk, deri_x)
	
	if np.allclose(deri_x, 0, atol=stopping_criteria):
		return iterations
	
	else:
		return newton_method(func, x + alpha * pk, hessian, iterations, epsilon, stopping_criteria)


def conjugate_gradient_1d(func, x, pk=None, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
		Univariate Case

		:param iterations: list containing all iterations we did already
		:param stopping_criteria: stopping_criteria for deriatives
		:param epsilon: epsilon to estimate deriative
		:param func: function to minimize
		:param x: starting point
		:return: array with steps
		"""
	# Append to list
	iterations.append(float(x))
	grad = estimate_deriative(func, x)
	# Initalize algorithm on first run
	if pk == None:
		pk = -grad
	
	# [1] Jonathan R. Shewchuk, 'An introduction to the conjugate-gradient method without the agonizing pain', pp.42-43.
	# For a quick algorithm I used the simple line search. Later on I will improve this very likely.
	alpha = calc_step_length(func, x, pk, grad)
	xk1 = x + alpha * pk
	
	gradxk1 = estimate_deriative(func, xk1)
	betak1 = (gradxk1 * gradxk1) / (grad * grad)
	pk1 = -gradxk1 + betak1 * pk
	
	if np.allclose(gradxk1, 0, stopping_criteria):
		iterations.append(float(xk1))
		return iterations
	else:
		return conjugate_gradient_1d(func, xk1, pk1, iterations, epsilon, stopping_criteria)


def conjugate_gradient(func, x, pk=None, iterations=[], epsilon=eps, stopping_criteria=eps):
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
	iterations.append(x)
	grad = estimate_deriative(func, x)
	# Initalize algorithm on first run
	if pk is None:
		pk = -grad
	
	# [1] Jonathan R. Shewchuk, 'An introduction to the conjugate-gradient method without the agonizing pain', pp.42-43.
	# For a quick algorithm I used the simple line search. Later on I will improve this very likely.
	alpha = calc_step_length(func, x, pk, grad)
	xk1 = x + alpha * pk
	
	gradxk1 = estimate_deriative(func, xk1)
	betak1 = (gradxk1 @ gradxk1) / (grad @ grad)
	pk1 = -gradxk1 + betak1 * pk
	
	if np.allclose(gradxk1, 0, atol=stopping_criteria):
		iterations.append(xk1)
		return iterations
	else:
		return conjugate_gradient(func, xk1, pk1, iterations, epsilon, stopping_criteria)


def quasi_newton_method_1d(func, x, inv_hessian=None, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param hessian: Hessian Matrix
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	
	if inv_hessian is None:
		inv_hessian = 1
	
	iterations.append(float(x))
	deri_x = estimate_deriative(func, x, epsilon)
	
	pk = -(inv_hessian * deri_x)
	
	alpha = calc_step_length(func, x, pk, deri_x)
	
	if np.allclose(pk, 0, stopping_criteria):
		return iterations
	
	else:
		xplus1 = x + alpha * pk
		
		deri_xplus1 = estimate_deriative(func, xplus1)
		sk = alpha * pk
		yk = deri_xplus1 - deri_x
		if yk != 0:
			inv_hessian = inv_hessian - inv_hessian * yk * yk * inv_hessian / (yk * inv_hessian * yk) + sk * sk / (
					yk * sk)
		
		return quasi_newton_method_1d(func, xplus1, inv_hessian, iterations, epsilon, stopping_criteria)


def quasi_newton_method(func, x, inv_hessian=None, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param hessian: Hessian Matrix
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	
	if inv_hessian is None:
		inv_hessian = np.diag(np.ones(shape=x.shape[0]))  # Taking multiply of the identity; ESTIMATE next time
	
	iterations.append(x)
	deri_x = estimate_deriative(func, x, epsilon)
	pk = -(inv_hessian @ deri_x)
	alpha = calc_step_length(func, x, pk, deri_x)
	
	if np.allclose(deri_x, 0, atol=stopping_criteria):
		return iterations
	
	else:
		xplus1 = x + alpha * pk
		
		deri_xplus1 = estimate_deriative(func, xplus1)
		sk = alpha * pk
		yk = deri_xplus1 - deri_x
		if yk.all != 0:
			rohk = 1 / (yk.T @ sk)
			
			inv_hessian = (np.diag(np.ones(shape=x.shape[0])) - (rohk * np.outer(sk, yk))) @ inv_hessian @ (
					np.diag(np.ones(shape=x.shape[0])) - rohk * np.outer(yk, sk)) + rohk * np.outer(sk, sk)  # pg 140
			
			return quasi_newton_method(func, xplus1, inv_hessian, iterations, epsilon, stopping_criteria)

