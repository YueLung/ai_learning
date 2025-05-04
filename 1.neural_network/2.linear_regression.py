import numpy as np
import matplotlib.pyplot as plt

x = np.array([3.75, 9.51, 7.32, 5.99, 1.56, 1.56, 0.58, 8.66, 6.01, 7.08,
     2.12, 9.70, 8.33, 2.90, 1.46, 6.12, 3.92, 1.83, 3.04, 5.25,
     4.32, 0.87, 3.26, 9.01, 2.66, 1.08, 3.36, 3.67, 6.92, 0.87])

y = np.array([10.08, 24.68, 19.20, 15.61, 5.15, 3.98, 3.11, 22.32, 16.69, 18.89,
     6.41, 25.08, 22.73, 8.48, 3.84, 17.34, 11.44, 6.45, 8.37, 14.22,
     11.60, 2.46, 9.73, 24.24, 8.42, 3.93, 10.44, 9.49, 19.94, 3.12])

def get_loss(x, y, w, b):
     y_hat = np.dot(x, w) + b
     loss = np.mean((y - y_hat) ** 2)
     # print(f'w = {w}, b = {b}')       
     return loss
 
# w = 2.52
# b = 1.24
w = 0
b = 0

print(f'init =  w = {w} b = {b}, loss = {get_loss(x, y, w, b)}') 

def update_weights(x, y, weight, bias, learning_rate):
    w_deriv = 0
    b_deriv = 0
    x_count = len(x)

    for i in range(x_count):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        w_deriv += -2 * x[i] * (y[i] - (weight * x[i] + bias))

        # -2(y - (mx + b))
        b_deriv += -2 * (y[i] - (weight * x[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (w_deriv / x_count) * learning_rate
    bias -= (b_deriv / x_count) * learning_rate

    return weight, bias

# train
loss_history = []
lr = 0.01
for i in range(5000): 
     w, b = update_weights(x, y, w, b, lr)
     loss_history.append(get_loss(x, y, w, b))

print(f'final =  w = {w} b = {b}, loss = {get_loss(x, y, w, b)}') 

x_line = np.linspace(min(x), max(x), 100)
y_line = w * x_line + b

plt.figure(figsize=(10, 5))  # 寬 10 吋, 高 5 吋

plt.subplot(1, 2, 1)
plt.title("y = wx + b")
plt.plot(x, y, 'o')
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_line, y_line, label=f'Best Fit: y = {w:.2f}x + {b:.2f}', color='red')

plt.subplot(1, 2, 2)
plt.title("loss history")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(loss_history)

# plt.legend()
# plt.grid(True)
plt.show()
