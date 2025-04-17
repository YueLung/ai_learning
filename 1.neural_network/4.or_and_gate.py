import numpy as np
import matplotlib.pyplot as plt


# 激活函數：Sigmoid & 導數
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# 成本函數（MSE）與導數
def cost(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cost_derivative(y_true, y_pred):
    return y_pred - y_true


# 神經網路訓練函數
def train_nn(X, y, epochs=1000, lr=0.1):
    n_samples, n_features = X.shape
    # 權重初始化（隨機）
    w = np.random.randn(n_features)
    b = 0

    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for xi, target in zip(X, y):
            # === forward propagation ===
            z = np.dot(xi, w) + b
            y_hat = sigmoid(z)

            # === cost ===
            loss = cost(target, y_hat)
            total_loss += loss

            # === backpropagation ===
            dcost_dy = cost_derivative(target, y_hat)
            dy_dz = sigmoid_derivative(z)

            dz_dw = xi
            dz_db = 1

            # 鏈式法則
            dloss_dw = dcost_dy * dy_dz * dz_dw
            dloss_db = dcost_dy * dy_dz * dz_db

            # 參數更新
            w -= lr * dloss_dw
            b -= lr * dloss_db

        losses.append(total_loss)

    return w, b, losses


# === 資料 ===
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# === 訓練 ===
w_and, b_and, losses_and = train_nn(X, y_and)
w_or, b_or, losses_or = train_nn(X, y_or)


def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


# 測試 AND Gate
print("=== AND Gate Prediction ===")
for xi, yi in zip(X, y_and):
    y_pred = predict(xi, w_and, b_and)
    print(
        f"Input: {xi}, Target: {yi}, Predicted: {y_pred:.4f}, Class: {int(y_pred >= 0.5)}"
    )

# 測試 OR Gate
print("\n=== OR Gate Prediction ===")
for xi, yi in zip(X, y_or):
    y_pred = predict(xi, w_or, b_or)
    print(
        f"Input: {xi}, Target: {yi}, Predicted: {y_pred:.4f}, Class: {int(y_pred >= 0.5)}"
    )


# === 繪圖 ===
plt.plot(losses_and, label="AND Gate Loss")
plt.plot(losses_or, label="OR Gate Loss")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Neural Network Training (AND vs OR)")
plt.legend()
plt.grid(True)
plt.show()
