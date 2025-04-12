import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Generate data
torch.manual_seed(42)
x_train = torch.linspace(-3, 3, 50).unsqueeze(1)
y_train = 2 * x_train - x_train**2 + 0.5 * x_train**3 + torch.randn_like(x_train) * 2
y_train = y_train.squeeze()  # (n,)

# Step 2: Define Polynomial Kernel Function
def polynomial_kernel(x1, x2, degree=3):
    return (x1 @ x2.T + 1) ** degree

# Step 3: Kernel Ridge Regression as nn.Module
class KernelRidgeRegression(nn.Module):
    def __init__(self, x_train, kernel_fn, lam=1.0):
        super().__init__()
        self.x_train = x_train.detach()
        self.kernel_fn = kernel_fn
        self.lam = lam

        # Dual coefficients α (one per training sample)
        self.alpha = nn.Parameter(torch.zeros(x_train.size(0)))

        # Precompute Gram matrix
        self.K = self.kernel_fn(self.x_train, self.x_train)  # shape: (n, n)

    def forward(self, x):
        # Compute kernel between x and training samples
        K_x = self.kernel_fn(x, self.x_train)  # shape: (m, n)
        return K_x @ self.alpha  # shape: (m,)

    def loss(self, y_pred, y_true):
        # Regularized loss in dual form
        fit_loss = torch.mean((y_pred - y_true) ** 2)
        reg_term = self.lam * self.alpha @ self.K @ self.alpha
        return fit_loss + reg_term

model = KernelRidgeRegression(x_train, kernel_fn=polynomial_kernel, lam=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = model.loss(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    x_test = torch.linspace(-3, 3, 200).unsqueeze(1)
    y_pred = model(x_test)

plt.scatter(x_train.numpy(), y_train.numpy(), label="Train Data", color='black')
plt.plot(x_test.numpy(), y_pred.numpy(), label="Kernel Ridge Fit", color='red')
plt.title("Kernel Ridge Regression (Polynomial Kernel)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
