import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Generate synthetic data
np.random.seed(0)
n = 300
x1_np = np.random.uniform(0, 1, size=n)
x2_np = np.random.uniform(0, 1, size=n)

# True underlying functions
def g1(x): return np.sin(2 * np.pi * x)
def g2(x): return (x - 0.5) ** 2

y_np = g1(x1_np) + g2(x2_np) + np.random.normal(0, 0.1, size=n)

# Step 2: Create cubic B-spline basis for each feature
B1 = dmatrix("bs(x1, df=6, degree=3, include_intercept=False)", {"x1": x1_np})
B2 = dmatrix("bs(x2, df=6, degree=3, include_intercept=False)", {"x2": x2_np})

# Convert to PyTorch tensors
X1 = torch.tensor(np.asarray(B1), dtype=torch.float32)
X2 = torch.tensor(np.asarray(B2), dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# Step 3: Define additive model
class AdditiveSplineModel(nn.Module):
    def __init__(self, num_basis1, num_basis2):
        super().__init__()
        self.beta1 = nn.Parameter(torch.randn(num_basis1, 1))
        self.beta2 = nn.Parameter(torch.randn(num_basis2, 1))
        self.bias = nn.Parameter(torch.tensor(0.0))  # Global intercept

    def forward(self, X1, X2):
        return self.bias + X1 @ self.beta1 + X2 @ self.beta2

model = AdditiveSplineModel(X1.shape[1], X2.shape[1])

# Step 4: Training
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(10000):
    model.train()
    y_pred = model(X1, X2)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 5: Plot fitted g1 and g2
x_plot = np.linspace(0, 1, 200)

# Design matrices for plotting
B1_plot = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x_plot})
B2_plot = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x_plot})

X1_plot = torch.tensor(np.asarray(B1_plot), dtype=torch.float32)
X2_plot = torch.tensor(np.asarray(B2_plot), dtype=torch.float32)

model.eval()
g1_hat = (X1_plot @ model.beta1).detach().numpy()
g2_hat = (X2_plot @ model.beta2).detach().numpy()

# True functions
g1_true = g1(x_plot)
g2_true = g2(x_plot)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(x_plot, g1_true, label='True $g_1$', linestyle='--')
axs[0].plot(x_plot, g1_hat, label='Estimated $g_1$')
axs[0].set_title('Component $g_1(x_1)$')
axs[0].legend()

axs[1].plot(x_plot, g2_true, label='True $g_2$', linestyle='--')
axs[1].plot(x_plot, g2_hat, label='Estimated $g_2$')
axs[1].set_title('Component $g_2(x_2)$')
axs[1].legend()

plt.tight_layout()
plt.show()
