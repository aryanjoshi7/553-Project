import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Generate synthetic data
np.random.seed(0)
n = 200
x_np = np.sort(np.random.uniform(0, 1, n))
y_np = np.sin(2 * np.pi * x_np) + np.random.normal(0, 0.1, size=n)

# Step 2: Generate B-spline basis (degree 3)
X_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x_np})
X_tensor = torch.tensor(np.asarray(X_design), dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# Step 3: Define the model with bias
class SplineModel(nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.beta = nn.Parameter(torch.randn(num_basis, 1))   # spline weights
        self.bias = nn.Parameter(torch.tensor(0.0))            # global intercept

    def forward(self, X_basis):
        return self.bias + X_basis @ self.beta

model = SplineModel(X_tensor.shape[1])

# Step 4: Train the model
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    model.train()
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 5: Plot prediction
x_test = np.linspace(0, 1, 200)
X_test_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x_test})
X_test_tensor = torch.tensor(np.asarray(X_test_design), dtype=torch.float32)

model.eval()
y_test_pred = model(X_test_tensor).detach().numpy()

plt.scatter(x_np, y_np, color='gray', alpha=0.5, label='Noisy data')
plt.plot(x_test, np.sin(2 * np.pi * x_test), label='True function', linestyle='--')
plt.plot(x_test, y_test_pred, label='Spline prediction (with bias)', linewidth=2)
plt.legend()
plt.title("Cubic B-Spline Regression in PyTorch (1 Feature)")
plt.show()