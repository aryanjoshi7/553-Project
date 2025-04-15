import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from csv_data_read import read_from_csv, normalize_meat_data, applyPCA

# 1. Generate synthetic data

meat = "meatspec.csv"

trim32 = "trim32_without_rownames"


labels, features = read_from_csv(meat, "fat")
labels, features = normalize_meat_data(labels, features)
n_features = features.shape[1]

y = labels.reshape(-1, 1)

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=.2, random_state=0)
# X_train, y_train = features, y
X_train, X_test = applyPCA(X_train, X_test)
formula = "bs(x, df=6, degree=3, include_intercept=True)"
# 3. Create spline basis design matrices (degree-3 B-spline)

X_train_designs = []
print("xtrain shape", X_train.shape, "features len", len(features))
print(y_train)
n_features = 30
# exit()
for i in range(n_features):
    X_train_designs.append(dmatrix(formula, {"x": X_train[:, i]}))

X_train_design = np.hstack(X_train_designs)

X_test_designs = []
for i in range(n_features):
    X_test_designs.append(dmatrix(formula, {"x": X_test[:, i]}))

X_test_designs = np.hstack(X_test_designs)

# # 4. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_design, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_designs, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 5. Create simple linear model with two separate sets of weights
class AdditiveModel(nn.Module):
    def __init__(self, n_features, n_basis):
        super().__init__()
        self.n_features = n_features
        self.n_basis = n_basis

        # One weight vector per feature
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, 1) * 0.01)
            for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # X shape: (batch_size, n_features * n_basis)
        outputs = []
        for i in range(self.n_features):
            start = i * self.n_basis
            end = (i + 1) * self.n_basis
            xi = X[:, start:end]
            wi = self.weights[i]
            outputs.append(xi @ wi)
        return sum(outputs) + self.bias

# print(X_train_design)
# exit()
# n_basis = X_train_design.shape[1]
n_basis = X_train_design.shape[1] // n_features
model = AdditiveModel(n_features,n_basis)

# 6. Training loop with Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)

_lambda_ = 0.5

def l2_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += (w ** 2).sum()
    l2 += (model.bias ** 2).sum()  # optional: remove if you don't want to regularize bias
    return _lambda_ * l2
def l1_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += torch.abs(w).sum()
    l2 += torch.abs(model.bias).sum()  # optional: remove if you don't want to regularize bias
    return _lambda_ * l2

def scad_penalty(model, lambda_, a=3.7):
    total_penalty = torch.tensor(0.0, device=model.weights[0].device)

    for w in model.weights:
        abs_w = torch.abs(w)

        # Case 1: |w| <= lambda
        penalty_1 = lambda_ * abs_w

        # Case 2: lambda < |w| <= a*lambda
        penalty_2 = ((-abs_w**2 + 2 * a * lambda_ * abs_w - lambda_**2) / (2 * (a - 1)))

        # Case 3: |w| > a*lambda
        penalty_3 = torch.full_like(abs_w, (a + 1) * lambda_**2 / 2)

        # Combine piecewise
        penalty = torch.where(
            abs_w <= lambda_,
            penalty_1,
            torch.where(
                abs_w <= a * lambda_,
                penalty_2,
                penalty_3
            )
        )

        total_penalty += penalty.sum()

    return total_penalty

# def mse_loss(model, y_pred, y_true):
#     return torch.mean((y_pred - y_true) ** 2)
# loss_fn = nn.MSELoss() 

for epoch in range(500):
    model.train()
    y_pred = model(X_train_tensor)
    loss = torch.mean((y_pred - y_train_tensor) ** 2) + .06*l1_penalty(model,_lambda_)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
# 7. Plot estimated functions
# def plot_partial_dependence(w, design_matrix_func, var_values, var_name):
#     grid = np.linspace(var_values.min(), var_values.max(), n)

#     grid_design = dmatrix(design_matrix_func, {"x": grid}, return_type='dataframe')
#     grid_tensor = torch.tensor(grid_design.values, dtype=torch.float32)
#     f_est = grid_tensor @ w.detach()
#     return grid, f_est.numpy()


y_pred = model(X_test_tensor)
loss = torch.mean((y_pred - y_test_tensor) ** 2)
print(loss)
# print(torch.topk(loss, 20, dim=0))
# for i, w in enumerate(model.weights):
#     print(f"Weight {i}:")
#     print(w.data)  # or w.detach().numpy() if you want NumPy arrays



# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# grid1, f1_est = plot_partial_dependence(model.w1, formula, x1, "x1")
# axs[0].plot(grid1, f1_est, color='blue')
# axs[0].plot(grid1, f1(grid1), color='red')
# # print(grid1.shape, y_1.shape)
# axs[0].scatter(x1, y_1, color="green")
# axs[0].set_title("Estimated f1(x1)")
# axs[0].set_xlabel("x1")
# axs[0].set_ylabel("Effect")
# axs[0].grid(True)

# grid2, f2_est = plot_partial_dependence(model.w2, formula, x2, "x2")
# axs[1].plot(grid2, f2_est, color='blue')
# axs[1].plot(grid2, f2(grid2), color='blue')
# axs[1].scatter(x2, y_2, color="green")
# axs[1].set_title("Estimated f2(x2)")
# axs[1].set_xlabel("x2")
# axs[1].set_ylabel("Effect")
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()




# tensor([[ 0.3412],
#         [-0.1026],
#         [ 3.6516],
#         [-3.5180],
#         [ 0.1202],
#         [ 2.0768],
#         [-1.6988],
#         [ 0.0662]], requires_grad=True) Parameter containing:
# tensor([[ 0.3488],
#         [-1.4044],
#         [ 1.6443],
#         [-0.1891],
#         [ 0.7715],
#         [ 0.3520],
#         [-0.2505],
#         [ 0.2919]], 