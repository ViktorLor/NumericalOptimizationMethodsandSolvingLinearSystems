"""
Author: Viktor Loreth
Date: 27.07.2022
Content: Simplex Method, Nocedal

Programming subject: Solving homework assignment Numerical Optimization at Kepler University
"""

import numpy as np


class Simplex:
    def __init__(self, A, b, c):
        # Concatenate slack variables
        A = np.concatenate((A, np.diag(v=np.ones(shape=A.shape[0]))), axis=1)
        self.A = A

        # Choose Initial B = slack variables
        self.B_i = list(range(A.shape[1] - A.shape[0], A.shape[1]))
        self.b = b
        c = np.concatenate((c, np.zeros(shape=A.shape[1] - c.shape[0])))
        self.c = c
        self.x = np.zeros_like(c)

    def function_value(self, x):
        return self.c @ self.x

    def feasible_start(self):
        # One could improve this method by deleting slack variables with E[i,i] = 1
        # This is not done due it costing time and it barely improves the time efficency.

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
        N_i = [_ for _ in range(0, self.c.shape[0]) if _ not in self.B_i]  # All indices not in B are in N
        self.x[self.B_i] = b
        while True:
            # Given step
            B = self.A[:, self.B_i]
            B_inverse = np.linalg.inv(B)
            for i, x_b_ele in enumerate(self.x[self.B_i]):
                if x_b_ele < 0:
                    raise ArithmeticError("X is negative, no solution to system")
                self.x[self.B_i[i]] = x_b_ele
            # self.x[N_i] = 0
            # Solve for Lambda and s_N
            lambda_ = B_inverse.T @ self.c[self.B_i]
            s_N = self.c[N_i] - self.A[:, N_i].T @ lambda_
            # check if all function values are positive
            if np.all(s_N >= 0):
                return self.x

            # find indices which are negative
            q = 0
            for i in range(s_N.shape[0]):
                if (s_N[i] < 0):
                    q = N_i[i]  # Entering indice
                    break

            # Find values which solve the equation
            d = B_inverse @ A[:, q]
            if np.all(d <= 0):
                print("Problem is unbounded")
                exit(0)

            # find maximum step size to minimize
            tmp = np.Inf
            p = 0
            for i in range(d.shape[0]):
                if d[i] == 0:
                    continue
                if tmp > self.x[self.B_i][i] / d[i]:
                    p = self.B_i[i]  # p = Minimizing column
                    tmp = self.x[self.B_i][i] / d[i]
                    x_qplus = tmp

            # Adjust x to new values
            self.x[self.B_i] = self.x[self.B_i] - d * x_qplus
            self.x[q] = x_qplus
            # Adjust basis

            a = self.B_i.index(p)

            self.B_i.remove(p)
            self.B_i.insert(a, q)

            a = N_i.index(q)
            N_i.remove(q)
            N_i.insert(a, p)
            self.stat()

    def stat(self):
        print(f"Current x = {self.x}")
        print(f"Function values = {self.function_value(self.x)}")


# Enter the Matrix A
A = np.asarray([[1, 1],
                [2, 0.5]])
# Define Boundary
b = np.asarray([5, 8])
# define function value
c = np.asarray([-4, -2])

my_simplex = Simplex(A, b, c)

# Finds a feasible starting point automatically
my_simplex.feasible_start()
# Runs a method with starting point x = 0
# solution = my_simplex.simplex_run()

my_simplex.stat()
