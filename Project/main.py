import numpy as np
from numerical_optimization.Project.Version2 import generating_functions as gf
import optimization_methods as om
import time
import matplotlib.pyplot as plt
import sys

dimensions = 2
residual_squares = False

if dimensions == 1:
	my_func = lambda x: x * +4 - 2 / 3 * x ** 3 - x ** 2 / 2 + 2 * x
	starting_point = np.array(-4, dtype=np.float32)
	gf.plot_1d_figure(-4, 4, my_func)
	x1 = -2
	print(f'Known solutions: {x1}')
	
	x = [x1]
	print('Steepest Descent')
	start_time = time.time()
	iterates = om.steepest_descent_1d(my_func, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	gf.print_statistics(x, my_func, iterates)
	
	print('Newton')
	start_time = time.time()
	iterates = om.newton_method_1d(my_func, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	gf.print_statistics(x, my_func, iterates)
	
	start_time = time.time()
	print('Conjugate Gradient')
	iterates = om.conjugate_gradient_1d(my_func, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	gf.print_statistics(x, my_func, iterates)
	
	start_time = time.time()
	print('Quasi-Newton')
	iterates = om.quasi_newton_method_1d(my_func, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	gf.print_statistics(x, my_func, iterates)

if dimensions == 2:
	
	my_func = gf.function_2()
	gf.plot_2d_figure(-5, 5, my_func)
	starting_point = np.array([-1.2,1], dtype=np.float32)
	known_minimzer = [[0,0]]
	
	#print('Steepest Descent')
	#start_time = time.time()
	#iterates = om.steepest_descent(my_func, starting_point)
	#print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	#gf.print_statistics(known_minimzer, my_func, iterates)
	
	#print('Conjugate Gradient')
	#start_time = time.time()
	#iterates = om.conjugate_gradient(my_func, starting_point)
	#print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	#gf.print_statistics(known_minimzer, my_func, iterates)
	
	#print('Quasi-Newton')
	#start_time = time.time()
	#iterates = om.quasi_newton_method(my_func, starting_point)
	#print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	#gf.print_statistics(known_minimzer, my_func, iterates)
	
	print('Newton Method')
	start_time = time.time()
	iterates = om.newton_method(my_func, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	gf.print_statistics(known_minimzer, my_func, iterates)
	


	


if residual_squares:
	sys.setrecursionlimit(1500)
	q = 1.5  # interval
	m = 50  # data samples
	n = 5  # degree
	rng = np.random.default_rng()
	m = rng.uniform(-q, q, m)  # points
	my_func, rjs = gf.least_squares(m, n)  # my_func returns func(x), [residual(x) for residuals]
	B = np.sin(m)
	starting_point = np.zeros(shape=n + 1)
	print('q = ', q)
	print('Degree = ', n)
	print('Data samples = ', len(m))
	print('function g(x) = sin(x)')
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.scatter(m, np.sin(m))
	X = np.linspace(-q, q, 1000)
	plt.plot(X, np.sin(X))
	
	print('Newton Method')
	start_time = time.time()
	iterates = om.newton_method(my_func, rjs, B, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	print(iterates[-1])
	print(my_func(iterates[-1]))
	f = gf.test_least_squares(iterates[-1])
	plt.plot(X, f(X), 'r')
	plt.show()
	
	print('Conjugate Gradient')
	start_time = time.time()
	iterates = om.conjugate_gradient(my_func, rjs, B, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	print(iterates[-1])
	print(my_func(iterates[-1]))
	
	print('Quasi-Newton')
	start_time = time.time()
	iterates = om.quasi_newton_method(my_func, rjs, B, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	print(iterates[-1])
	print(my_func(iterates[-1]))
	
	print('Steepest Descent')
	start_time = time.time()
	iterates = om.steepest_descent(my_func, rjs, B, starting_point)
	print("Time taken: %s seconds\\\\" % (time.time() - start_time))
	print(iterates[-1])
	print(my_func(iterates[-1]))
