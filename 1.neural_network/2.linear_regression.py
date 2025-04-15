import numpy as np
import matplotlib.pyplot as plt

x = np.array([3.75, 9.51, 7.32, 5.99, 1.56, 1.56, 0.58, 8.66, 6.01, 7.08,
     2.12, 9.70, 8.33, 2.90, 1.46, 6.12, 3.92, 1.83, 3.04, 5.25,
     4.32, 0.87, 3.26, 9.01, 2.66, 1.08, 3.36, 3.67, 6.92, 0.87])

y = np.array([10.08, 24.68, 19.20, 15.61, 5.15, 3.98, 3.11, 22.32, 16.69, 18.89,
     6.41, 25.08, 22.73, 8.48, 3.84, 17.34, 11.44, 6.45, 8.37, 14.22,
     11.60, 2.46, 9.73, 24.24, 8.42, 3.93, 10.44, 9.49, 19.94, 3.12])

w = 3
b = -1

y_hat = np.dot(x, w) + b
loss = np.mean((y - y_hat) ** 2)

print(loss) 

plt.plot(x, y_hat)
# # plt.plot(y, label="y")
plt.scatter(x, y)
# plt.title("y=xw+b")
plt.xlabel("x")
plt.ylabel("y")
# # plt.legend()
# # plt.grid(True)
plt.show()
