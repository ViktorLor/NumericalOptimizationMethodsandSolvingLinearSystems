"""
Active Set Method for Conveq QP
Taken from Numerical Optimization pg 472.

Author: Viktor Loreth
Date: 19.06.2022
"""

import numpy as np

def KKT_solver(G, g, A=None, h=None):
	if A is not None:
		blocked_left = np.block(
			[[G, A.T], [A, np.zeros((G.shape[0] + A.shape[0] - A.T.shape[0], G.shape[1] + A.T.shape[1] - A.shape[1]))]])
		blocked_right = np.block([g, h]).T
		solution = np.linalg.solve(blocked_left, blocked_right)
		p = -solution[:c.shape[0]]
		lamda = solution[c.shape[0]:]
	
	else:
		p = -np.linalg.solve(G, g)
		lamda = None
	
	return p, lamda


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
	pk, lamda = KKT_solver(G, G @ xk + c, A[W, :], A[W, :] @ xk - b[W])
	
	print("---")
	print("pk = ", pk)
	print("xk = ", xk)
	

	
	if np.all(pk== 0):
		if np.all(lamda >= 0):
			print(f"Solution = {xk}\n lambda = {lamda}")
			print(exit(1))
		else:
			print("lamda = ", lamda)
			j = np.argmin(lamda)
			
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
