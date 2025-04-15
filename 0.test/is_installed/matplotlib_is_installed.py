import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y, marker='o')
plt.title("Simple Line Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()