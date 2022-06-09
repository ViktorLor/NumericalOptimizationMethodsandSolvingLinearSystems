import numpy as np
import generating_functions as gf
import optimization as om
import timeit
from matplotlib import pyplot as plt

### Rosenbrock and CO
if False:
	
	my_func = gf.function_2()
	
	gf.plot_2d_figure(-10, 10, my_func)
	
	starting_points = [np.array([-0.2, 1.2], dtype=float), np.array([3.8, 0.1], dtype=float),
	                   np.array([0, 0], dtype=float), np.array([-1, 0], dtype=float), np.array([0, -1], dtype=float)]
	
	for starting_point in starting_points[0:3]:
		
		print("StartingPoint: ", starting_point,"\\\\")
		Optimization = om.Optimization(starting_point, my_func)
		solution = Optimization.conjugate_gradient()
		print("Iterations: ", len(solution),"\\\\")
		print("Found Solution: ", solution[-1],"\\\\")
		print("Distance to Solution [0,1]: ",solution[-1] - np.array([0,1]),"\\\\")
		print("Distance to Solution: [4,0]", solution[-1] - np.array([4,0]), "\\\\")
		print("-------------","\\\\")

### Residuals

if True:
	q = 2.5  # interval
	m = 100  # data samples
	n = 5 # degree
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
	plt.show()
	####Preparation
	
	OptimizationR = om.ResidualOptimization(starting_point, my_func)
	solution = OptimizationR.steepest_descent(rjs, B)
	print("Steepest Descent")
	print("Iterations: ", len(solution), "\\\\")
	print("Found Solution: ", solution[-1], "\\\\")
	print("-------------", "\\\\")

	f = gf.test_least_squares(solution[-1])
	plt.plot(X, f(X), 'r')
	func = gf.taylor_sin(20)
	plt.plot(X, func(X), 'b')
	plt.scatter(m, np.sin(m))
	plt.show()

	OptimizationR = om.ResidualOptimization(starting_point, my_func)
	solution = OptimizationR.newton_method(rjs,B)
	print("Newton\\\\")
	print("Iterations: ", len(solution), "\\\\")
	print("Found Solution: ", solution[-1], "\\\\")
	print("-------------", "\\\\")

	f = gf.test_least_squares(solution[-1])
	plt.plot(X, f(X), 'r')
	func = gf.taylor_sin(20)
	plt.plot(X, func(X),'b')
	plt.scatter(m, np.sin(m))
	plt.show()

	OptimizationR = om.ResidualOptimization(starting_point, my_func)
	solution = OptimizationR.quasi_newton_method(rjs,B)
	print("Quasi_newton\\\\")
	print("Iterations: ", len(solution), "\\\\")
	print("Found Solution: ", solution[-1], "\\\\")
	print("-------------", "\\\\")

	f = gf.test_least_squares(solution[-1])
	plt.plot(X, f(X), 'r')
	func = gf.taylor_sin(20)
	plt.plot(X, func(X),'b')
	plt.scatter(m, np.sin(m))
	plt.show()

	OptimizationR = om.ResidualOptimization(starting_point, my_func)
	solution = OptimizationR.conjugate_gradient(rjs,B)
	print("Conjugate_gradient\\\\")
	print("Iterations: ", len(solution), "\\\\")
	print("Found Solution: ", solution[-1], "\\\\")
	print("-------------", "\\\\")

	f = gf.test_least_squares(solution[-1])
	plt.plot(X, f(X), 'r')
	func = gf.taylor_sin(20)
	plt.plot(X, func(X),'b')
	plt.scatter(m, np.sin(m))
	plt.show()