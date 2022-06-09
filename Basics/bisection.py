# Implements the bisection method to find a minimum

def f(x):
	return x ** 5 - x ** 4 - 1


def bisection_method(a, b, k):
	c = (a + b) / 2
	if abs(f(c)) < k:
		return c
	
	if f(c) * f(a) <= 0:
		return bisection_method(a, c, k)
	elif f(b) * f(c) < 0:
		return bisection_method(c, b, k)


# you need to check before if f(a) < 0 and f(b) > 0
x = bisection_method(0.3, 1.9, 0.1)
print(x)
