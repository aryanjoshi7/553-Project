import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from patsy import dmatrix
import matplotlib.pyplot as plt

# Step 1: Generate data
np.random.seed(0)
n = 200
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

def f1(x): return np.sin(x)
def f2(x): return np.log1p(x)
y = f1(x1) + f2(x2) + np.random.normal(0, 0.3, size=n)

# Step 2: Create basis design matrix (B-splines for each feature)
X1_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x1})
X2_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x2})
X_np = np.hstack([X1_design, X2_design])

# Step 3: Convert to torch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
print(X_tensor.shape)
exit()
# Step 4: Define linear model for additive spline weights
model = nn.Linear(X_tensor.shape[1], 1, bias=True)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# Step 5: Train the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

# Step 6: Visualize prediction
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

plt.figure(figsize=(10, 5))
plt.scatter(x1, y, alpha=0.4, label="True y")
plt.scatter(x1, y_pred, alpha=0.4, label="Predicted y (via splines)")
plt.title("Additive Model Fit with PyTorch + Adam")
plt.xlabel("x1 (note: only shown for one axis)")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
