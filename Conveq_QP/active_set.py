"""
Active Set Method for Conveq QP
Taken from Numerical Optimization pg 472.

Author: Viktor Loreth
Date: 19.06.2022
"""

import numpy as np

# Set feasible starting point xk:
from Tools.scripts.var_access_benchmark import A

xk = np.array([0, 0])

# constraints:
A = np.array([[1, 1],
              [1, 0],
              [0, 1]])
b = np.array([3, 0, 0])
G = np.array([[2, 0],
              [0, 2]])
c = np.array([-6, -4])

constraints_ind = np.array([0, 1, 2])
W = [0, 1]  # working set indices

lambda_i = np.array([0, 0, 0])

Gminus = np.linalg.inv(G)
while (True):
	
	# Find a way to calculate pk
	# min
	h = A[W]
	g = c + G @ xk
	pk = [1.5,1.5]
	
	
	print("---")
	print("pk = ", pk)
	print("xk = ", xk)
	if np.all(pk - xk == 0):
		
		lambda_i = np.linalg.lstsq(A.T[W], G @ xk + c, rcond=None)[0]
		if np.all(lambda_i >= 0):
			print(f"Solution = {xk}\n lambda = {lambda_i}")
			print(exit(1))
		else:
			print("lambda = ", lambda_i)
			j = np.argmin(lambda_i)
			
			del W[j]
	
	else:
		indice = [x for x in constraints_ind if x not in W]
		
		alphak = 1
		for i in indice:
			if A[i] @ pk < 0:
				alphak = min(alphak, (b[i] - A[i].flatten() @ xk) / (A[i].flatten() @ pk))
				W.append(i)
				break
		xk = xk + alphak * pk
