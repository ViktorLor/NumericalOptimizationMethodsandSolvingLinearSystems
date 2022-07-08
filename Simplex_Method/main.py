import numpy as np

### 2 Problems with 5-10 variables
# Enter the Matrix A
A = np.asarray([[0, 1, 1, 0],
                [3, 1, 0, 1]])

# Calculate Problem size:
n = A.shape[0]
# Choose Initial B = slack variables
B_i = [2, 3]
# N = inverse of B

# Choose b as boundary
b = np.asarray([3, 4])
# all variables and their respective values in their functions, including slack variables
c = np.asarray([-3, -3, 0, 0])


# function to optimize:
def f(x):
    return - 3 * x[0] - 3 * x[1]


print('A:\n', A)
print('b:', b)
print('Function to minimize', c)


def simplex_method(B_i, A, b, c):
    x_N = 0
    x = np.zeros(shape=(c.shape[0]))
    N_i = [x for x in range(0, c.shape[0]) if x not in B_i]
    while True:

        B = A[:, B_i]
        B_inverse = np.linalg.inv(B)
        x_B = B_inverse @ b
        for i, x_b_ele in enumerate(x_B):
            if x_b_ele < 0:
                raise ArithmeticError("X is negative, no solution to system")
            x[B_i[i]] = x_b_ele
        x[N_i] = 0

        print("x_B:", x_B)
        lambda_ = B_inverse.T @ c[B_i]
        s_N = c[N_i] - A[:, N_i].T @ lambda_
        print("s_N: ", s_N)
        print("lambda_: ", lambda_)
        if np.all(s_N >= 0):
            return x

        q = 0
        for i in range(s_N.shape[0]):
            if (s_N[i] < 0):
                q = i
                break
        d = B_inverse @ A[:, q]

        if np.all(d <= 0):
            print("Problem is unbounded")
            exit(0)

        tmp = np.Inf
        p = 0
        for i in range(d.shape[0]):
            if (d[i] == 0):
                continue
            if tmp > x_B[i] / d[i]:
                p = i
                tmp = x_B[i] / d[i]
                x_qplus = tmp

        q = N_i[q]
        x[q] = x_qplus
        # x_Bplus = x_B - d * x_qplus
        #
        # x_nplus = np.zeros(shape=A.shape[0])
        # x_nplus[p] = x_qplus

        k = B_i[p]
        a = B_i.index(B_i[p])
        B_i.remove(B_i[p])
        B_i.insert(a, q)

        a = N_i.index(q)

        N_i.remove(q)
        N_i.insert(a, k)

        print("x: ", x)
        print("B_i: ", B_i)
        print("N_i: ", N_i)
        print('f(x) = ', f(x[0:c.shape[0]]))
        print("Next step -----")


x = simplex_method(B_i, A, b, c)
print("Solution x: ", x)
print("f(x)", f(x))
