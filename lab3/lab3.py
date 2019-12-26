import numpy as np
from numpy import linalg as la
from scipy.optimize import root
import random
'''
f_0 = 0.5 * x^T * A * x + b^T * x
f = ||x - x_0 || - z <= 0
f^2 =  ||x - x_0 ||^2 - z^2 <= 0
L(x, y) = 0.5 * x^T * A * x + b^T * x + y * (||x - x_0 ||^2 - z^2 )
1. y > 0
Ax + b + 2y(x - x_0) = 0
||x - x_0||^2 = r^2
x = (A + 2Iy)^(-1)(2yx_0 - b)
||x - x_0||^2 = r^2
'''


def f_0(A, b, x):
    return (x @ A @ x.transpose() + b @ x.transpose())[0, 0]


def f(x_0, r, x):
    return la.norm(x - x_0, 2) - r


def F(y):
    return pow(la.norm(x_(y) - x_0, 2), 2) - z * z


def x_(y):
    return (2 * y * x_0 - b) @ la.inv(A + 2 * y * np.eye(n))


n = 6
A = np.random.randint(1, 5, (n, n))
A = A @ A.transpose()
b = np.random.rand(1, n)
x_0 = np.random.rand(1, n)
z = random.uniform(0.2, 10.0)

A_1 = la.inv(A)
x_1 = -(b @ A_1)
print('arg min f_0 = ', x_1)
print('min f = ', f_0(A, b, x_1))

if f(x_0, z, x_1) <= 0:
    print('also min with the condition f(x) <= 0')
    exit(0)

sol = root(F, 0)
print(sol.message)
print(sol.x)
y = sol.x[0]
x_ans = x_(sol.x)
if f(x_0, z, x_1) <= 0 or y <= 0:
    print('condition don\'t resolve')
    exit(0)

print('min with the condition f(x) <= 0')
print(x_ans)
print('f = ', f_0(A, b, x_ans))
