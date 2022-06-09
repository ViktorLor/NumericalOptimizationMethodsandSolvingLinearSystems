import math

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from pylab import figure, cm


def polynomial_1d(my_zeros: np.ndarray):
	"""
	:param my_list: zeros in list
	:return: polynomial with zero position at my_list
	"""
	
	def poly(x):
		func = 1
		for zeros in my_zeros:
			func *= (zeros - x)
			"if function is 1D"
		return func
	
	return poly


def taylor_sin(degree):
	def func(x):
		
		k = 0
		for n in range(1, degree+1):
			k += (-1) ** (n + 1) * (x ** (2 * n - 1)) / float(math.factorial(2 * n - 1))
		return k
	
	return func

def polynomial_nd(my_zeros: np.ndarray):
	"""
	:param my_list: zeros in list
	:return: polynomial with zero position at my_list
	"""
	
	def poly(x):
		func = 1
		for j, row in enumerate(my_zeros):
			for zeros in row:
				func *= (zeros - x[j])
				"if function is 1D"
		
		return func
	
	return poly

def sin_nd(d=2):
	"""
	:param my_list: zeros in list
	:return: polynomial with zero position at my_list
	"""
	
	def my_func(x):
		func = 1
		for j in range(2):
			func += np.sin(x[j]) + np.cos(x[j])
		return func
	
	return my_func

def least_squares(ajs, degree=10):
	def my_func(x):
		func = 0
		
		for aj in ajs:
			
			tmp_func = 0
			for j in range(degree + 1):
				tmp_func += x[j] * aj ** j
			tmp_func -= np.sin(aj)
			
			tmp_func = tmp_func ** 2
			func += tmp_func
		
		return func / 2
	
	def rjs(x):
		rj = []
		for aj in ajs:
			tmp_func = 0
			for j in range(degree + 1):
				tmp_func += x[j] * aj ** j
			tmp_func -= np.sin(aj)
			rj.append(tmp_func)
		return np.asarray(rj)
	
	return my_func, rjs

def test_least_squares(x):
	def my_func(aj):
		func = 0
		
		for j in range(len(x)):
			func += x[j] * aj ** j
		
		return func
	
	return my_func

def plot_1d_figure(x1, x2, my_func):
	X = np.linspace(x1, x2, 300)
	# We use this function to check if we are right aswell
	deri = nd.Derivative(my_func)
	
	fig, ax = plt.subplots()
	plt.plot(X, my_func(X))
	plt.plot(X, deri(X))
	ax.axhline(y=0, xmin=0.0, xmax=1.0, color='r')
	plt.legend(['Function', 'Deriative'])
	plt.show()

def plot_2d_figure(x1_, x2_, my_func):
	x1, x2 = np.meshgrid(np.arange(x1_, x2_, 0.1), np.arange(x1_, x2_, 0.1))
	
	Z = my_func([x1, x2])
	
	plt.imshow(Z, extent=[x1_, x2_, x1_, x2_], cmap=cm.terrain, origin='lower')
	plt.colorbar()
	plt.axis('scaled')
	plt.colorbar()
	plt.show()

def print_statistics(known_minimizers, func, iterates):
	print(f'Number of Iterates: {len(iterates)}\\\\\nFound solution: {iterates[-1]}\\\\')
	
	print(f'Iterates: {list(np.round(iterate, 2) for iterate in iterates[0:10])}\\\\')
	x = 10000
	for solution in known_minimizers:
		
		if np.sum(abs(iterates[-1] - solution)) < np.sum(abs(np.abs(x))):
			x = iterates[-1] - solution
	
	print(f'found solution - real solution = {x}\\\\\nf(x) = {func(iterates[-1])}\\\\')

def rosenbrock_function():
	def func(x):
		return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
	
	return func

def function_2():
	def func(x):
		return 150 * (x[1] * x[0]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2
	
	return func
