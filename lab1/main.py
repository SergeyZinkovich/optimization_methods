import math

import numpy as np
import sympy as sp

n = 14
np.random.seed(5)
X = sp.MatrixSymbol('x', n, 1)
A = np.random.rand(n, n) * 10
A = sp.Matrix(A.dot(A.transpose()))
b = sp.Matrix(np.random.rand(n, 1))

f = (0.5 * (X.T * sp.Matrix(A) * X))[0] + b.dot(sp.Matrix(X))

d_f = []
for i in range(n):
    d_f.append(f.diff(X[i]))
d_f = sp.Matrix(d_f)

dd_f = []
for i in range(n):
    dd_f.append([])
    for j in range(n):
        dd_f[i].append(f.diff(X[i]).diff(X[j]))
dd_f = sp.Matrix(dd_f)


def newton_method():
    x_k = sp.Matrix(np.random.rand(n))
    for i in range(100):
        x_k = x_k - (dd_f.subs(X, x_k) ** -1) * d_f.subs(X, x_k)
    return x_k

def grad_method():
    lamb = 0.001
    eps = 1e-5
    x_k = sp.Matrix(np.random.rand(n))
    i = 0
    while True:
        i += 1
        x_k_p1 = x_k - lamb * d_f.subs(X, x_k)
        magnitude = math.sqrt(sum(i**2 for i in (x_k_p1 - x_k)))
        if magnitude < eps:
            print(i)
            return x_k_p1
        x_k = x_k_p1



print("Newton")
newton_ans = newton_method()
print(newton_ans)
print(f.subs(X, newton_ans))
print("Gradient")
grad_ans = grad_method()
print(grad_ans)
print(f.subs(X, grad_ans))
