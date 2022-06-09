import numpy as np
import numdifftools as nd

eps = 1 / np.power(10, 5)


def estimate_deriative(func, rjs, x, epsilon=eps):
	# nd case
	jacobian = calc_jacobian(func, rjs, x)
	
	return jacobian.T @ rjs(x)


def estimate_hessian(func, rjs, x, epsilon=eps):
	# nd case
	jacobian = calc_jacobian(func, rjs, x)
	
	return jacobian.T @ jacobian


def calc_jacobian(func, rjs, x, epsilon=eps):
	m = len(rjs(x))
	jacobian = np.ndarray(shape=(m, x.shape[0]))
	
	for i, xn in enumerate(x):
		epsilon_vector = np.zeros_like(x)
		epsilon_vector[i] = epsilon
		
		jacobian[:, i] = (rjs(x + epsilon_vector) - rjs(x - epsilon_vector)) / (2 * epsilon)
	return jacobian


def calc_step_length(func, rjs, B, x, pk, deri_x: np.ndarray):
	jacobian = calc_jacobian(func, rjs, x)
	H = estimate_hessian(func, rjs, x)

	alpha = ((pk.T @ jacobian.T) @ B - pk.T @ H @ x - x.T @ H @ pk + B.T @ jacobian @ pk) / (2 * pk.T @ H @ pk)
	return alpha


def steepest_descent(func, rjs, B, x, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	iterations.append(x)
	
	deri_x = estimate_deriative(func, rjs, x, epsilon)
	pk = -deri_x
	
	alpha = calc_step_length(func, rjs, B, x, pk, deri_x)
	
	if np.allclose(func(x), 0, atol=1 / np.power(10, 2)):  # Using a different rule here.
		return iterations
	
	else:
		return steepest_descent(func, rjs, B, x + alpha * pk, iterations, epsilon, stopping_criteria)


def newton_method(func, rjs, B, x, iterations=None, epsilon=eps, stopping_criteria=eps):
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
	
	hessian = estimate_hessian(func, rjs, x)
	
	iterations.append(x)
	deri_x = estimate_deriative(func, rjs, x)
	inv_hessian = np.linalg.inv(hessian)
	
	pk = -(inv_hessian @ deri_x)
	
	alpha = calc_step_length(func, rjs, B, x, pk, deri_x)
	
	if np.allclose(func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
		return iterations
	
	else:
		return newton_method(func, rjs, B, x + alpha * pk, iterations, epsilon, stopping_criteria)


def conjugate_gradient(func, rjs, B, x, pk=None, iterations=[], epsilon=eps, stopping_criteria=eps):
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
	grad = estimate_deriative(func, rjs, x)
	# Initalize algorithm on first run
	if pk is None:
		pk = -grad
	
	alpha = calc_step_length(func, rjs, B, x, pk, grad)
	xk1 = x + alpha * pk
	
	gradxk1 = estimate_deriative(func, rjs, xk1)
	betak1 = (gradxk1 @ gradxk1) / (grad @ grad)
	pk1 = -gradxk1 + betak1 * pk
	
	if np.allclose(func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
		iterations.append(xk1)
		return iterations
	else:
		return conjugate_gradient(func, rjs, B, xk1, pk1, iterations, epsilon, stopping_criteria)


def quasi_newton_method(func, rjs, B, x, inv_hessian=None, iterations=[], epsilon=eps, stopping_criteria=eps):
	"""
	:param inv_hessian:
	:param iterations: list containing all iterations we did already
	:param stopping_criteria: stopping_criteria for deriatives
	:param epsilon: epsilon to estimate deriative
	:param func: function to minimize
	:param x: starting point
	:return: array with steps
	"""
	
	if inv_hessian is None:
		inv_hessian = np.linalg.inv(
			estimate_hessian(func, rjs, x))  # Taking multiply of the identity; ESTIMATE next time
	
	iterations.append(x)
	deri_x = estimate_deriative(func, rjs, x)
	pk = -(inv_hessian @ deri_x)
	alpha = calc_step_length(func, rjs, B, x, pk, deri_x)
	
	if np.allclose(func(x), 0, atol=1 / np.power(10, 3)):  # Using a different rule here.
		return iterations
	
	else:
		xplus1 = x + alpha * pk
		
		deri_xplus1 = estimate_deriative(func, rjs, xplus1)
		sk = alpha * pk
		yk = deri_xplus1 - deri_x
		if yk.all != 0:
			rohk = 1 / (yk.T @ sk)
			
			inv_hessian = (np.diag(np.ones(shape=x.shape[0])) - (rohk * np.outer(sk, yk))) @ inv_hessian @ (
					np.diag(np.ones(shape=x.shape[0])) - rohk * np.outer(yk, sk)) + rohk * np.outer(sk, sk)  # pg 140
			
			return quasi_newton_method(func, rjs, B, xplus1, inv_hessian, iterations, epsilon, stopping_criteria)
