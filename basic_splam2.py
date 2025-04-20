from sklearn.model_selection import KFold

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from patsy import dmatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from csv_data_read import read_from_csv, normalize_data, applyPCA
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
        # self.linear_weights = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, 1) * 0.01)
            for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # X shape: (batch_size, n_features * n_basis)
        # print(X.shape)
        linear_output = 0
        # X_first_30 = X[:, :self.n_features]
        # linear_output = X_first_30 @ self.linear_weights
        # X = X[:, self.n_features:]

        # print(X.shape)

        outputs = []
        for i in range(self.n_features):
            start = i * self.n_basis
            end = (i + 1) * self.n_basis
            xi = X[:, start:end]
            wi = self.weights[i]
            outputs.append(xi @ wi)
        outputs.append(linear_output)
        result = sum(outputs) + self.bias
        # print("result", result, result.shape)

        return result
    
class SPLAMAdditiveModel(nn.Module):
    def __init__(self, n_features, n_basis):
        super().__init__()
        self.n_features = n_features
        self.n_basis = n_basis

        # One weight vector per feature
        self.linear_weights = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, 1) * 0.01)
            for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # X shape: (batch_size, n_features * n_basis)
        # print(X.shape)
        print(X.shape, self.n_basis)

        linear_output = 0
        X_first_30 = X[:, :self.n_features]
        linear_output = X_first_30 @ self.linear_weights
        X = X[:, self.n_features:]

        print(X.shape, self.n_basis)

        outputs = []
        for i in range(self.n_features):
            start = i * self.n_basis
            end = (i + 1) * self.n_basis
            xi = X[:, start:end]
            wi = self.weights[i]
            outputs.append(xi @ wi)
        outputs.append(linear_output)
        result = sum(outputs) + self.bias
        # print("result", result, result.shape)

        return result

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
def scad_function(r, parameters: Parameters, isLinear=False):
    if isLinear:
        r = r * parameters.splam_alpha
    else:
        r = r * (1 - parameters.splam_alpha)

    if r <= parameters.scad_lambda:
        penalty = parameters.scad_lambda * r
    elif r <= parameters.scad_alpha * parameters.scad_lambda:
        penalty = (-r**2 + 2 * parameters.scad_alpha * parameters.scad_lambda * r - parameters.scad_lambda**2) / (2 * (parameters.scad_alpha - 1))
    else:
        penalty = (parameters.scad_alpha + 1) * parameters.scad_lambda**2 / 2

    return penalty

def group_scad_penalty(model: AdditiveModel, parameters: Parameters):
    total_penalty = torch.tensor(0.0, device=model.weights[0].device)
    
    for w in model.weights:  # shape (n_basis, 1)

        r = torch.norm(w)  # scalar: ||β_j||₂
        penalty = scad_function(r, parameters, isLinear=False)
        total_penalty += penalty

    r = torch.norm(model.linear_weights)  # scalar: ||β_j||₂
    total_penalty += scad_function(r, parameters, isLinear=True)

    return total_penalty

def mse_loss(y_pred, y_true, parameters: Parameters):
    return torch.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred, y_true, parameters: Parameters):
    return torch.mean(torch.abs(y_pred - y_true))

# read this function for mspe
def quantile_loss(y_pred, y_true, parameters: Parameters):
    error = y_true - y_pred
    # print()
    return torch.mean(torch.maximum(parameters.quantile_tau * error, (parameters.quantile_tau - 1) * error))
# implement qb here
def quantile_bias_loss(y_pred, y_true, parameters: Parameters):
    tau = parameters.quantile_tau
    indicators = (y_true <= y_pred).float()
    return torch.abs(torch.mean(indicators - tau))

def fit(X, y, n_features, n_basis, loss_fn, penalty_fn, parameters : Parameters):
    if parameters.splam:
        model = SPLAMAdditiveModel(n_features, X.shape[1] - n_features)
    else:
        model = AdditiveModel(n_features, n_basis)

    # 6. Training loop with Adam
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # def mse_loss(model, y_pred, y_true):
    #     return torch.mean((y_pred - y_true) ** 2)
    # loss_fn = nn.MSELoss() 
    just_loss = []
    penalties = []
    loss_penalty = []
    for epoch in range(16000):
        model.train()
        y_pred = model(X)
        penalty = parameters.lambda_ * penalty_fn(model, parameters)
        loss = loss_fn(y_pred, y, parameters) 
        loss += penalty
        # print(epoch, loss.item(), penalty.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        just_loss.append(loss.item() - penalty.item())
        penalties.append(penalty.item())
        loss_penalty.append(loss.item())
    print("training loss", round(loss.item(), 6), round(penalty.item(), 6))
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

def plot_fit(model: AdditiveModel, X_train, y_train, formula):


    # Design matrices for plotting
    B1_plot = dmatrix(formula, {"x": X_train})

    X1_plot = torch.tensor(np.asarray(B1_plot), dtype=torch.float32)

    model.eval()
    g1_hat = model(X1_plot).detach().numpy()

    # True functions
    # g1_true = g1(x_plot)

    # Plotting
    # plt.plot(x_plot, g1_true, label='True $g_1$', linestyle='--')
    plt.plot(X_train, g1_hat, label='Estimated $g_1$')
    plt.scatter(X_train, y_train)
    # plt.scatter(x_plot, g1_true)

    plt.legend()
    plt.show()

def generate_basis_tensor(X_train, y_train, parameters: Parameters):
    n_features = X_train.shape[1]
    # Create spline basis design matrices (degree-3 B-spline)
    X_train_designs = []
    for i in range(n_features):
        X_train_designs.append(dmatrix(parameters.formula, {"x": X_train[:, i]}))

    X_train_design = np.hstack(X_train_designs)
    
    if parameters.splam:
        X_train_design = np.hstack([X_train, X_train_design]) 
    # print(X_train_design.shape)


    # # 4. Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_design).float()
    y_train_tensor = y_train.clone().detach().float()
    return X_train_tensor, y_train_tensor

from collections import defaultdict

mse_losses = []
mae_losses = []
mcpe_losses = [] # list of losses per quantile, list is length of folds
qb_losses = defaultdict(list)
def cross_validate(X_full, y_full, penalty_fn, loss_fn, parameters: Parameters, k=20, random_state=1):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    n_features = X_full.shape[1]
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        # making the folds
        X_train_fold = X_full[train_idx].clone().detach().float()
        y_train_fold = y_full[train_idx].clone().detach().float()
        X_test_fold = X_full[val_idx].clone().detach().float()
        y_test_fold = y_full[val_idx].clone().detach().float()

        # print(X_train_fold)
        # normalizing before pca
        # normalized_y_train, normalized_X_train = normalize_data(y_train_fold, X_train_fold)
        # normalized_y_test, normalized_X_test = normalize_data(y_test_fold, X_test_fold)

        # PCA and defining the "formula", pca normalizes it again
        # print(normalized_X_train.shape)
        # print(normalized_X_test.shape)
        X_train, X_test = X_train_fold, X_test_fold
        # X_train, X_test = applyPCA(normalized_X_train, normalized_X_test)
        print(X_full.shape)


        # # 4. Convert to PyTorch tensors
        X_train_tensor, y_train_tensor = generate_basis_tensor(X_train, y_train_fold, parameters)
        y_train_tensor = y_train_fold.clone().detach().float()

        X_test_tensor, y_test_tensor = generate_basis_tensor(X_test, y_test_fold, parameters)


        n_basis = X_train_tensor.shape[1] // n_features

        # fitting the model here
        for quantile in parameters.quantile_tau_list:
            parameters.quantile_tau = quantile
            # print(X_train_tensor.shape)
            # exit()
            model = fit(X_train_tensor, y_train_tensor, n_features, n_basis, loss_fn, penalty_fn, parameters)

            y_pred = model(X_test_tensor)
            if quantile == .5:
                mse_losses.append(mse_loss(y_pred, y_test_tensor, parameters).item())
                mae_losses.append(mae_loss(y_pred, y_test_tensor, parameters).item())
            mcpe_losses.append(quantile_loss(y_pred, y_test_tensor, parameters).item())
            qb_losses[quantile].append(quantile_bias_loss(y_pred, y_test_tensor, parameters).item())
            # print(X_train_tensor.shape)
            # print(y_train_tensor.shape)
            # plot_fit(model, X_full, y_full, formula)
            return model
    

    # print(f"\nAverage Validation Loss over {k} folds: {avg_val_loss:.4f}")
    # return avg_val_loss

quantiles = [.5]
# labels, features = read_from_csv(meat, "fat")
# labels = labels.reshape(-1, 1)


def g1(x): return .5*x


# 1. Generate synthetic data
np.random.seed(0)
n = 200
x1 = np.random.uniform(-20, 20, n)



# additive nonlinear regression with basis splines of quartic degree

y_1 = g1(x1) #+ np.random.normal(0, 0.3, size=n)
y = y_1

y = y.reshape(-1, 1)

features, labels = x1, y
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
features = features.view(n, 1)

# print(features.shape)
# exit()
spline_formula = "bs(x, df=6, degree=3, include_intercept=True)"
splam_formula = "bs(x, df=6, degree=3, include_intercept=True)"

model_parameters = Parameters(quantile_tau_list=quantiles, lambda_=.025, scad_alpha=3.7, scad_lambda=.4, splam_alpha=.4, formula=spline_formula)
splam_parameters = Parameters(quantile_tau_list=quantiles, lambda_=.025, scad_alpha=3.7, scad_lambda=.4, splam_alpha=.4, formula=splam_formula, splam=True)

for i in range(1):
    model = cross_validate(features, labels, group_scad_penalty, mae_loss, parameters=model_parameters, random_state=i, k=10)
# for i in range(1):
#     splam_model = cross_validate(features, labels, l2_penalty, mae_loss, parameters=splam_parameters, random_state=i, k=10)

# features, inds = torch.sort(features, dim=0)
# labels = labels[inds.squeeze()]
# print(features)
# X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, model_parameters)
# model_pred = model(X_train_tensor).detach().numpy()

# # Plotting
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# axs[0].plot(features, labels, label='True $g_1$', linestyle='--')
# axs[0].plot(features, model_pred, label='Estimated $g_1$')
# axs[0].set_title('Cubic Splines')
# axs[0].legend()



# X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, splam_parameters)
# splam_model_pred = splam_model(X_train_tensor).detach().numpy()

# axs[1].plot(features, labels, label='True $g_2$', linestyle='--')
# axs[1].plot(features, splam_model_pred, label='Estimated $g_2$')
# axs[1].set_title('Cubic splines with linear feature')
# axs[1].legend()

# plt.tight_layout()
# plt.show()


# Avg mse 0.044464402832090855
# Avg mae 0.16067576706409453
# Avg mcpe 0.08033788353204727

# print("mse_losses: ", mse_losses)
# print("mae_losses: ", mae_losses)
# print("mcpe_losses: ", mcpe_losses)
# print("qb_losses: ", qb_losses)

per_quantile_qb_avg = [sum(qb_losses[q]) for q in quantiles]
print("----------------------------")
print("Avg mse", sum(mse_losses)/len(mse_losses))
print("Avg mae", sum(mae_losses)/len(mae_losses))
print("Avg mcpe", sum(mcpe_losses)/len(mcpe_losses))
print("Avg qb", sum(per_quantile_qb_avg)/len(per_quantile_qb_avg))
print(per_quantile_qb_avg)
