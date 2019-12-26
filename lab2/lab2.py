import numpy as np
from numpy import linalg as la
import random
'''
f_0 = 0.5 * x^T * A * x + b^T * x
f_x = c^T * x + d <= 0
L(x, y) = f_0 + y *f =  0.5 * x^T * A * x + b^T * x + y * (c^T * x + d)
1. Ax + b + y * c = 0
2. y (c^T * x + d) = 0
1. y = 0
x = -A^-1 * b
2. y > 0
y = (d - c^T * A^-1 * b) / (c^T * A^-1 * c)
x = -A^-1 (b + y * c)
'''

n = 6
A = np.random.rand(n, n)
A = A @ A.transpose()
b = np.random.rand(1, n).transpose()
c = np.random.rand(1, n).transpose()
d = random.uniform(1.0, 100.0)

A_1 = la.inv(A)
x_1 = -(A_1 @ b)
print('arg min f_0 = \n', x_1)

y = (d - (c.transpose() @ A_1 @ b)[0, 0]) / (c.transpose() @ A_1 @ c)[0, 0]
x = -(A_1 @ (b + y * c))
print('min with the condition f(x) <= 0')
print(x)
