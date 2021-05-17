import numpy as np
import efemarai as ef

a = np.arange(16)

print(a)

a_2d = a.reshape(4, 4)
print(a_2d)

a_3d = a.reshape(2, 2, 4)
print(a_3d)


ef.inspect(a_2d)
ef.inspect(a_3d)


b = np.random.randn(4, 4)

ef.inspect(b)

addition = a_2d + b
ef.inspect(addition)

exit(0)
melementwise = a_2d * b
ef.inspect(melementwise, name="Elementwise")

mdot = np.dot(a_2d, b)
ef.inspect(mdot, name="Dot product")

X = np.array([1, 4, 3])
Y = np.array([2, 3, 2])

mcross = np.cross(X, Y)
print(mcross)

X = np.random.randn(25, 25, 25)
ef.inspect(X, name="Large tensor")

ReLU_X = np.maximum(0, X)
ef.inspect(ReLU_X, name="ReLU")
