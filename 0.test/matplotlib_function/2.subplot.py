import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(8, 6))

# plt.subplot(行數, 列數, 第幾個圖)

# 第一個子圖：sin(x)
plt.subplot(2, 1, 1)
plt.plot(x, y1, color='blue')
plt.title('sin(x)')

# 第二個子圖：cos(x)
plt.subplot(2, 1, 2)
plt.plot(x, y2, color='green')
plt.title('cos(x)')

plt.tight_layout()  # 自動調整排版避免重疊
plt.show()
