import numpy as np

print("Python list operations")

a = [1, 2, 3]
b = [4, 5, 6]

print("a+b = ", a + b)

try:
    print("a*b = ", a * b)
except TypeError:
    print("a*b has no meaning for Python lists")

print()
print("numpy array operations: ")

a = np.array(a)
b = np.array(b)

print("a+b =", a + b)
print("a*b =", a * b)
print()

a = np.array([[1, 2], [3, 4]])
print("a =", a)
print("a.sum(axis=0) =", a.sum(axis=0))
print("a.sum(axis=1) =", a.sum(axis=1))
print()

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print("a+b =", a + b)
print()
