# run 1:
# zeros = np.array([0, -3, 2, 4])
# my_func = gf.polynomial_1d(zeros)
# starting_point = np.array(2, dtype=np.float32)
# gf.plot_1d_figure(-3.5, 4.5, my_func)
# print('Known solutions: x1 = -1,935362601; x2 = 3,223663518')
# x1 = -1.935362601
# x2 = 3.223663518

# run 2:
# zeros = np.array([0, -3, 2, 4])
# my_func_tmp = gf.polynomial_1d(zeros)
# my_func = lambda x: -np.cos(my_func_tmp(x))
# starting_point = np.array(2, dtype=np.float32)
# gf.plot_1d_figure(-3.5, 4.5, my_func)
# print('Known solutions: x1 = -1,935362601; x2 = 3,223663518')
# x1 = -1.935362601
# x2 = 3.223663518

# run 3:
# my_func = lambda x: x * +4 - 2 / 3 * x ** 3 - x ** 2 / 2 + 2 * x
# starting_point = np.array(-4, dtype=np.float32)
# gf.plot_1d_figure(-4, 4, my_func)
# x1 = -2
# print(f'Known solutions: {x1}')

# run 4
# zeros2d = np.array([[1, -1, 2], [2, 1, -1.1]])
# my_func = gf.polynomial_nd(zeros2d)
# gf.plot_2d_figure(-2, 2, my_func)
# starting_point = np.array([1.2, 1.3], dtype=np.float32)
# known_minimzer = [[1, 1], [-0.21524794, 1.54681343]]

# run 5
# my_func = gf.sin_nd()
# gf.plot_2d_figure(-5, 5, my_func)
# starting_point = np.array([0, 0], dtype=np.float32)
# known_minimzer = [[-2.3561899, -2.3561899], [3.92699556, 3.92699556]]

# run 6
# q = 1  # interval
# m = 100  # data samples
# n = 5  # degree
