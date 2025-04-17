from sklearn.model_selection import KFold

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from csv_data_read import read_from_csv, normalize_meat_data, applyPCA
from parameters import Parameters
# 1. Generate synthetic data

meat = "meatspec.csv"

trim32 = "trim32_without_rownames"


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
def l2_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += (w ** 2).sum()
    # l2 += (model.bias ** 2).sum()  # optional: remove if you don't want to regularize bias
    return l2

def l1_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += torch.abs(w).sum()
    # l2 += torch.abs(model.bias).sum()  # optional: remove if you don't want to regularize bias
    return l2

def group_l2_penalty(model, _lambda_):
    penalty = torch.tensor(0.0, device=model.weights[0].device)
    for w in model.weights:  # shape (K, 1)
        penalty += torch.sqrt((w ** 2).sum())
    return _lambda_ * penalty

def group_l1_penalty(model, parameters: Parameters):
    penalty = torch.tensor(0.0, device=model.weights[0].device)
    for w in model.weights:  # w.shape = (n_basis, 1)
        penalty += torch.sqrt((w ** 2).sum())  # ||β_j||₂
    return  penalty

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

def group_scad_penalty(model, lambda_, a=3.7):
    total_penalty = torch.tensor(0.0, device=model.weights[0].device)

    for w in model.weights:  # shape (n_basis, 1)
        r = torch.norm(w)  # scalar: ||β_j||₂

        if r <= lambda_:
            penalty = lambda_ * r
        elif r <= a * lambda_:
            penalty = (-r**2 + 2 * a * lambda_ * r - lambda_**2) / (2 * (a - 1))
        else:
            penalty = (a + 1) * lambda_**2 / 2

        total_penalty += penalty

    return total_penalty

def mse_loss(y_pred, y_true, parameters: Parameters):
    return torch.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred, y_true, parameters: Parameters):
    return torch.mean(torch.abs(y_pred - y_true))

# read this function for mspe
def quantile_loss(y_pred, y_true, parameters: Parameters):
    error = y_true - y_pred
    return torch.mean(torch.maximum(parameters.quantile_tau * error, (parameters.quantile_tau - 1) * error))
# implement qb here
def qb():
    pass

def fit(X, y, n_features, n_basis, loss_fn, penalty_fn, parameters : Parameters):
    n_basis = X.shape[1] // n_features
    model = AdditiveModel(n_features, n_basis)

    # 6. Training loop with Adam
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # def mse_loss(model, y_pred, y_true):
    #     return torch.mean((y_pred - y_true) ** 2)
    # loss_fn = nn.MSELoss() 
    just_loss = []
    penalties = []
    loss_penalty = []
    for epoch in range(500):
        model.train()
        y_pred = model(X)
        penalty = parameters.lambda_ * penalty_fn(model, parameters)
        loss = loss_fn(y_pred, y, parameters) 
        loss += penalty
        print(epoch, loss.item(), penalty.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        just_loss.append(loss.item() - penalty.item())
        penalties.append(penalty.item())
        loss_penalty.append(loss.item())
    # 7. Plot estimated functions
    # def plot_partial_dependence(w, design_matrix_func, var_values, var_name):
    #     grid = np.linspace(var_values.min(), var_values.max(), n)

    #     grid_design = dmatrix(design_matrix_func, {"x": grid}, return_type='dataframe')
    #     grid_tensor = torch.tensor(grid_design.values, dtype=torch.float32)
    #     f_est = grid_tensor @ w.detach()
    #     return grid, f_est.numpy()


    # y_pred = model(X_test_tensor)
    # mae = mae_loss(y_pred, y_test_tensor, parameters)
    # mse = mse_loss(y_pred, y_test_tensor, parameters)
    # print("test loss", mae.item(), mse.item())
    # plt.plot(loss_penalty) 
    # plt.show()
    return model
# print(torch.sort(loss_n, dim=0))
# fit(mse_loss, group_l1_penalty, parameters=Parameters(quantile_tau=.5, lambda_=.03), )
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

from collections import defaultdict


def cross_validate(X_full, y_full, penalty_fn, loss_fn, parameters: Parameters, k=5):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_losses = []
    mae_losses = []
    mcpe_losses = [] # list of losses per quantile, list is length of folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        # making the folds
        X_train_fold, y_train_fold = torch.tensor(X_full[train_idx], dtype=torch.float32), torch.tensor(y_full[train_idx], dtype=torch.float32)
        X_test_fold, y_test_fold = torch.tensor(X_full[val_idx], dtype=torch.float32), torch.tensor(y_full[val_idx], dtype=torch.float32)
        # print(X_train_fold)
        # normalizing before pca
        normalized_y_train, normalized_X_train = normalize_meat_data(y_train_fold, X_train_fold)
        normalized_y_test, normalized_X_test = normalize_meat_data(y_test_fold, X_test_fold)

        # PCA and defining the "formula", pca normalizes it again
        print(normalized_X_train.shape)
        print(normalized_X_test.shape)
        
        X_train, X_test = applyPCA(normalized_X_train, normalized_X_test)
        n_features = 30
        
        # Create spline basis design matrices (degree-3 B-spline)
        formula = "bs(x, df=6, degree=3, include_intercept=True) + x"
        X_train_designs = []
        for i in range(n_features):
            X_train_designs.append(dmatrix(formula, {"x": X_train[:, i]}))

        X_train_design = np.hstack(X_train_designs)

        X_test_designs = []
        for i in range(n_features):
            X_test_designs.append(dmatrix(formula, {"x": X_test[:, i]}))

        X_test_designs = np.hstack(X_test_designs)

        # # 4. Convert to PyTorch tensors
        X_train_tensor, y_train_tensor = torch.tensor(X_train_design, dtype=torch.float32), torch.tensor(normalized_y_train, dtype=torch.float32)
        X_test_tensor, y_test_tensor = torch.tensor(X_test_designs, dtype=torch.float32), torch.tensor(normalized_y_test, dtype=torch.float32)

        n_basis = X_train_tensor.shape[1] // n_features

        # fitting the model here
        model = fit(X_train_tensor, y_train_tensor, n_features, n_basis, loss_fn, penalty_fn, parameters)

        y_pred = model(X_test_tensor)
        mse_losses.append(mse_loss(y_pred, y_test_tensor, parameters).item())
        mae_losses.append(mae_loss(y_pred, y_test_tensor, parameters).item())
        mcpe_losses.append(quantile_loss(y_pred, y_test_tensor, parameters).item())


    # avg_val_loss = np.mean(val_losses)
    
    print("mse_losses: ", mse_losses)
    print("mae_losses: ", mae_losses)
    print("mcpe_losses: ", mcpe_losses)
    
    # print(f"\nAverage Validation Loss over {k} folds: {avg_val_loss:.4f}")
    # return avg_val_loss


labels, features = read_from_csv(meat, "fat")
labels = labels.reshape(-1, 1)
cross_validate(features, labels, group_l1_penalty, mae_loss, parameters=Parameters(quantile_tau=.5, lambda_=.02))
