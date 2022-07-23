import numpy as np


class QP():

    def __init__(self, G, c, A, b):
        dim = len(c)
        self.G = G
        self.c = c
        self.A = A
        self.b = b
        self.x = np.zeros(dim)
        self.x_old = np.zeros(dim)
        self.p = np.zeros(dim)
        self.p_old = np.zeros(dim)
        self.Wk = []
        self.Wk_old = []
        self.active = np.zeros(len(b), dtype=bool)
        self.lm = np.zeros(len(b))
        self.alpha = 0
        self.num_c = len(b)
        self.tol = 1e-3

    def set_start(self, x):
        self.x = x
        for i, a in enumerate(self.A):
            if a @ self.x == self.b[i]:
                self.active[i] = True
                break

    def xkp1(self):
        return self.x + self.p * self.alpha

    def calc_p(self):
        # if len(self.A[self.active]) != 1:
        #    return np.array([0., 0.])
        gk = self.G @ self.x + self.c
        # g= gk +
        w_size = np.sum(self.active)
        # p = (-1) * np.linalg.solve(self.G, gk)
        A = np.bmat([[self.G, -self.A[self.active].T], [self.A[self.active], np.zeros(shape=(w_size, w_size))]])
        b = np.pad(-gk, (0, w_size))
        lagrange = np.linalg.solve(A, b)
        if w_size == 0:
            self.p = lagrange
        else:
            self.p = lagrange[:-w_size]
            self.lm[self.active] = lagrange[-w_size:]
        # r1 = np.concatenate(self.G, A.T, axis=1)
        # r2 = np.concatenate(self.A, np.zeros((2,2)), axis=1)

        # L, B, P = ldl(np.concatenate(r1, r2, axis=0))

        # z = np.linalg.solve(L, P.T @ np.concatenate())

    def get_alpha(self):
        # if np.any(self.p >= 0):
        #    return 1
        mini = np.ones(self.num_c)
        for i in range(self.num_c):
            if self.active[i]:
                continue
            if self.A[i] @ self.p >= 0:
                continue
            mini[i] = (self.b[i] - self.A[i] @ self.x) / (self.A[i] @ self.p)
        return min(1, np.min(mini))

    def solve(self):
        xx = []
        for k in range(100):

            xx.append(self.x)
            self.calc_p()
            x = self.x
            pk = self.p

            # print(f"pk: {pk}")

            if np.allclose(self.p, 0, rtol=self.tol):
                # lagrange multipliers already calculated in calc_p
                if np.all(self.lm[self.active] >= 0):
                    print(f"Found solution in {k + 1} iterations!")
                    return xx
                else:
                    j = np.argmin(self.lm)
                    self.x_old = self.x
                    self.active[j] = False
                    self.lm[j] = 0.0
            else:
                self.alpha = self.get_alpha()
                self.x = self.xkp1()
                # print(f"a: {self.alpha}, x: {self.x}")
                for i, con in enumerate(self.active):
                    if con:
                        continue
                    else:
                        if self.A[i] @ self.x <= self.b[i]:
                            self.active[i] = True
                            break

        print("Run out of iterations")
        return xx

    def get_f(self, x=None):
        if x is None:
            x = self.x
        return 0.5 * x.T @ self.G @ x + x.T @ self.c

    def check_minimum(self, p=False):
        sol = np.around(self.x, decimals=4)
        num = 10
        f_sol = self.get_f(sol)
        if p:
            print("Solution: ", f_sol)
        for i in range(num):
            dx = self.tol * np.array([np.sin(2 * np.pi * i / 10), np.cos(2 * np.pi * i / 10)])
            # print(self.A @ (sol + dx) - self.b)
            if np.any(self.A @ (sol + dx) - self.b < 0):  # not within boundaries
                continue
            local_sol = self.get_f(sol + dx)
            if p:
                print("Solution: ", local_sol)
            if local_sol < f_sol:
                print("NOT a Minimum!")
                return
        print(f"{sol} is indeed a Minimum")


if __name__ == "__main__":
    print("START!")
    G = np.array([[2, -3], [-3, 6]])
    c = np.array([-2, -2])
    A = np.array([[-4, 3], [5, 2]])
    b = np.array([8, 1])
    print("G = \n ", G)
    print("c = ", c)
    print("A = \n", A)
    print("b = ", b)
    op = QP(G, c, A, b)
    # x_0 = np.array([1.5, 0.5])
    x_0 = np.array([0.5, 0.5], dtype=float)

    print("x_0: ", x_0)
    op.set_start(x_0)
    xs = op.solve()

    print(xs)
    print(xs[-1])

    op.check_minimum()
