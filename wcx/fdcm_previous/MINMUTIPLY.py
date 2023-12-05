import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 创建函数，用于生成不同属于一个平面的100个离散点
def not_all_in_plane(a, b, c):
    x = np.random.uniform(-10, 10, size=100)
    y = np.random.uniform(-10, 10, size=100)
    z = (a * x + b * y + c) + np.random.normal(-1, 1, size=100)
    return x, y, z


# 调用函数，生成离散点
x2, y2, z2 = not_all_in_plane(2, 5, 6)

# 创建系数矩阵A
A = np.zeros((3, 3))
for i in range(0, 100):
    A[0, 0] = A[0, 0] + x2[i] ** 2
    A[0, 1] = A[0, 1] + x2[i] * y2[i]
    A[0, 2] = A[0, 2] + x2[i]
    A[1, 0] = A[0, 1]
    A[1, 1] = A[1, 1] + y2[i] ** 2
    A[1, 2] = A[1, 2] + y2[i]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = 100
# print(A)

# 创建b
b = np.zeros((3, 1))
for i in range(0, 100):
    b[0, 0] = b[0, 0] + x2[i] * z2[i]
    b[1, 0] = b[1, 0] + y2[i] * z2[i]
    b[2, 0] = b[2, 0] + z2[i]
# print(b)

# 求解X
A_inv = np.linalg.inv(A)
X = np.dot(A_inv, b)
print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))

# 计算方差
R = 0
for i in range(0, 100):
    R = R + (X[0, 0] * x2[i] + X[1, 0] * y2[i] + X[2, 0] - z2[i]) ** 2
print('方差为：%.*f' % (3, R))

# 展示图像
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.scatter(x2, y2, z2, c='r', marker='o')
x_p = np.linspace(-10, 10, 100)
y_p = np.linspace(-10, 10, 100)
x_p, y_p = np.meshgrid(x_p, y_p)
z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
ax1.plot_wireframe(x_p, y_p, z_p, rstride=10, cstride=10)
plt.show()