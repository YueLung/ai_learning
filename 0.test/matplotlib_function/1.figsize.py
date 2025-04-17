import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))  # 寬 10 吋, 高 5 吋
plt.plot(x, y)
plt.title("正弦波")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
