# import torch
# import matplotlib.pyplot as plt

# # Create synthetic data
# torch.manual_seed(0)
# n = 100
# x = torch.linspace(-3, 3, n).unsqueeze(1)
# y = 10*x + torch.randn_like(x) * 2  # cubic function with noise

# def polynomial_features(x, degree=3):
#     return torch.cat([x**i for i in range(1, degree+1)], dim=1)

# X_poly = polynomial_features(x, degree=3)  # Shape: (n, 3)

# lambda_reg = 10.0
# X = X_poly
# y = y.squeeze()

# # Closed-form Ridge solution
# I = torch.eye(X.shape[1])
# theta = torch.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y

# # Predict on training data
# y_pred = X @ theta

# # Plot results
# plt.scatter(x.numpy(), y.numpy(), label='Noisy Data')
# plt.plot(x.numpy(), y_pred.detach().numpy(), color='red', label='Ridge Prediction')
# plt.title('Kernel Ridge Regression with Polynomial Kernel (Explicit Features)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

# import torch
# import matplotlib.pyplot as plt

# # Step 1: Generate synthetic data
# torch.manual_seed(42)
# n = 100
# x = torch.linspace(-3, 3, n).unsqueeze(1)
# y_true = 2 * x - x**2 + 0.5 * x**3
# y = y_true + torch.randn_like(x) * 2  # add noise
# y = y.squeeze()  # shape: (n,)

# # Step 2: Expand features to include x, x^2, x^3
# def polynomial_features(x, degree=3):
#     return torch.cat([x**i for i in range(1, degree+1)], dim=1)

# X = polynomial_features(x, degree=3)  # shape: (n, 3)

# # Step 3: Initialize parameters carefully
# w = torch.zeros(3, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# # Hyperparameters
# learning_rate = 1e-3
# lambda_reg = 1.0
# epochs = 1000

# # Step 4: Training loop
# for epoch in range(epochs):
#     # Forward pass
#     y_pred = X @ w + b

#     # Compute loss: MSE + L2 regularization
#     mse_loss = torch.mean((y_pred - y)**2)
#     ridge_penalty = lambda_reg * torch.sum(w**2)
#     loss = mse_loss + ridge_penalty

#     # Backward pass
#     loss.backward()

#     # Gradient descent update
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#         b -= learning_rate * b.grad
#         w.grad.zero_()
#         b.grad.zero_()

#     # Print progress every 100 steps
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# # Step 5: Plot predictions
# with torch.no_grad():
#     y_pred = X @ w + b

# plt.scatter(x.numpy(), y.numpy(), label='Noisy data')
# plt.plot(x.numpy(), y_pred.numpy(), color='red', label='Model prediction')
# plt.title('Ridge Regression (Gradient Descent)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

import torch
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
torch.manual_seed(42)
n = 100
x = torch.linspace(-20, 20, n).unsqueeze(1)
y_true = x
y = y_true + torch.randn_like(x) * 0  # add noise
y = y.squeeze()  # shape: (n,)

# Step 2: Polynomial feature expansion
def polynomial_features(x, degree=3):
    return torch.cat([x**i for i in range(1, degree+1)], dim=1)

X = polynomial_features(x, degree=3)  # shape: (n, 3)

# Step 3: Initialize parameters and enable gradient tracking
w = torch.zeros(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Step 4: Set up Adam optimizer
optimizer = torch.optim.Adam([w, b], lr=0.001)
lambda_reg = 5
epochs = 30000

# Step 5: Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    y_pred = X @ w + b

    # Loss = MSE + Ridge penalty
    mse_loss = torch.mean((y_pred - y)**2)
    ridge_penalty = lambda_reg * torch.sum(w**2) * 0
    loss = mse_loss + ridge_penalty

    # Backward pass and optimization
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # Progress print
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Step 6: Plot predictions
with torch.no_grad():
    y_pred = X @ w + b
print("w:", w)
print("b:", b)
plt.scatter(x.numpy(), y.numpy(), label='Noisy Data')
plt.plot(x.numpy(), y_pred.numpy(), color='red', label='Adam Prediction')
plt.title('Ridge Regression with Adam Optimizer')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()