import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
np.random.seed(0)
n = 200
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

def f1(x): return np.sin(x) + 1
def f2(x): return np.log(x) + 1
y = f1(x1) + f2(x2) + np.random.normal(0, 0.3, size=n)

X = np.vstack([x1, x2]).T
y = y.reshape(-1, 1)

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
formula = "bs(x, df=6, degree=4, include_intercept=True) + x"
# 3. Create spline basis design matrices (degree-3 B-spline)
X1_train_design = dmatrix(formula, {"x": X_train[:, 0]})
X2_train_design = dmatrix(formula, {"x": X_train[:, 1]})
X_train_design = np.hstack([X1_train_design, X2_train_design])

X1_test_design = dmatrix(formula, {"x": X_test[:, 0]})
X2_test_design = dmatrix(formula, {"x": X_test[:, 1]})
X_test_design = np.hstack([X1_test_design, X2_test_design])

# 4. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_design, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_design, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 5. Create simple linear model with two separate sets of weights
class AdditiveModel(nn.Module):
    def __init__(self, n_basis):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_basis, 1) * 0.01)
        self.w2 = nn.Parameter(torch.randn(n_basis, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        x1 = X[:, :n_basis]
        x2 = X[:, n_basis:]
        return x1 @ self.w1 + x2 @ self.w2 + self.bias

n_basis = X1_train_design.shape[1]
model = AdditiveModel(n_basis)

# 6. Training loop with Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(8000):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 7. Plot estimated functions
def plot_partial_dependence(w, design_matrix_func, var_values, var_name):
    grid = np.linspace(var_values.min(), var_values.max(), 100)
    grid_design = dmatrix(design_matrix_func, {"x": grid}, return_type='dataframe')
    grid_tensor = torch.tensor(grid_design.values, dtype=torch.float32)
    f_est = grid_tensor @ w.detach()
    return grid, f_est.numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

grid1, f1_est = plot_partial_dependence(model.w1, formula, x1, "x1")
axs[0].plot(grid1, f1_est, color='blue')
axs[0].plot(grid1, f1(grid1), color='blue')
axs[0].set_title("Estimated f1(x1)")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("Effect")
axs[0].grid(True)

grid2, f2_est = plot_partial_dependence(model.w2, formula, x2, "x2")
axs[1].plot(grid2, f2_est, color='blue')
axs[1].plot(grid2, f2(grid2), color='blue')
axs[1].set_title("Estimated f2(x2)")
axs[1].set_xlabel("x2")
axs[1].set_ylabel("Effect")
axs[1].grid(True)

plt.tight_layout()
plt.show()
