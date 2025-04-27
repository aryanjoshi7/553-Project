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

        outputs = []
        for i in range(self.n_features):
            start = i * self.n_basis
            end = (i + 1) * self.n_basis
            xi = X[:, start:end]
            wi = self.weights[i]
            # print(xi.shape, wi.shape)
            outputs.append(xi @ wi)
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
        # print(X.shape, self.n_basis)

        linear_output = 0
        X_first_30 = X[:, :self.n_features]
        linear_output = X_first_30 @ self.linear_weights
        X = X[:, self.n_features:]

        # print(X.shape, self.n_basis)

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

def l2_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += (w ** 2).sum()
    # l2 += (model.bias ** 2).sum()  # optional: remove if you don't want to regularize bias
    return l2

def group_l2_penalty(model, parameters: Parameters):
    penalty = torch.tensor(0.0, device=model.weights[0].device)
    for w in model.weights:  # shape (K, 1)
        penalty += torch.sqrt((w ** 2).sum())
    return penalty

def l1_penalty(model, _lambda_):
    l2 = torch.tensor(0.0, device=model.weights[0].device)  # keep on same device
    for w in model.weights:
        l2 += torch.abs(w).sum()
    # l2 += torch.abs(model.bias).sum()  # optional: remove if you don't want to regularize bias
    return l2


def group_l1_penalty(model, parameters: Parameters):
    penalty = torch.tensor(0.0, device=model.weights[0].device)
    for w in model.weights:  # w.shape = (n_basis, 1)
        penalty += torch.sqrt((w ** 2).sum())  # ||β_j||₂
    return  penalty
scad_norm = 1

def scad_penalty(model, parameters: Parameters):
    lambda_ = parameters.scad_lambda
    a = parameters.scad_alpha
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
def scad_penalty_splam(model, parameters: Parameters):
    lambda_ = parameters.scad_lambda
    a = parameters.scad_alpha
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

    return (parameters.splam_alpha)*total_penalty
def scad_function(r, parameters: Parameters, isLinear=False):
    # if isLinear:
    #     r = r * parameters.splam_alpha
    # else:
    #     r = r * (1 - parameters.splam_alpha)

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
        r = torch.norm(w, p=2)  # scalar: ||β_j||₂
        penalty = scad_function(r, parameters, isLinear=False)
        if parameters.splam:
            penalty = (1 - parameters.splam_alpha)*penalty
        total_penalty += penalty

    if parameters.splam:
        r = torch.norm(model.linear_weights, p = 1)  # scalar: ||β_j||₂
        if parameters.splam:
            penalty = (parameters.splam_alpha)*penalty
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
    total = torch.sum(indicators)
    total = total / y_true.shape[0]
    total = total - tau
    total = torch.abs(total)
    # print("qb",total, torch.abs(torch.mean(indicators - tau)))
    return torch.abs(torch.mean(indicators - tau))
def new_quantile_bias_loss(y_pred, y_true, parameters: Parameters):
    tau = parameters.quantile_tau
    indicators = (y_true <= y_pred).float()
    return torch.abs(torch.mean(indicators - tau))
def fit(X, y, n_features, n_basis, loss_fn, penalty_fn, parameters : Parameters):
    if parameters.splam:
        # print(n_features, X.shape[1] - n_features)
        # exit()
        model = SPLAMAdditiveModel(n_features, (X.shape[1] - n_features)//n_features)
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
    num_epochs = 5000
    if basic:
        num_epochs = 10000
    prev_loss = 100
    count_no_improve = 0
    epochs = 0
    for _ in range(num_epochs):
        model.train()
        y_pred = model(X)
        penalty = parameters.lambda_ * penalty_fn(model, parameters)
        loss = loss_fn(y_pred, y, parameters) 
        loss += penalty
        if prev_loss < loss.item():
            count_no_improve += 1
            if count_no_improve > 10:
                break
        prev_loss = loss.item()

        # print(epoch, loss.item(), penalty.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        just_loss.append(loss.item() - penalty.item())
        penalties.append(penalty.item())
        loss_penalty.append(loss.item())
        epochs += 1
        # print(loss.item(), penalty.item(), epoch)
    print("training loss", round(loss.item(), 6), round(penalty.item(), 6), epochs)
    return model

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
training_loss = []
training_penalty = []

qb_losses = defaultdict(list)
def cross_validate(X_full, y_full, penalty_fn, loss_fn, parameters: Parameters, k=20, random_state=1,):
    n_features = X_full.shape[1]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        # making the folds
        X_train_fold = X_full[train_idx].clone().detach().float()
        y_train_fold = y_full[train_idx].clone().detach().float()
        X_test_fold = X_full[val_idx].clone().detach().float()
        y_test_fold = y_full[val_idx].clone().detach().float()

        # PCA and defining the "formula", pca normalizes it again
        # print(normalized_X_train.shape)
        # print(normalized_X_test.shape)
        X_train, X_test = X_train_fold, X_test_fold
        if not basic:
            y_train_fold, normalized_X_train = normalize_data(y_train_fold, X_train_fold)
            y_test_fold, normalized_X_test = normalize_data(y_test_fold, X_test_fold)
            X_train, X_test = applyPCA(normalized_X_train, normalized_X_test)
            n_features = X_train.shape[1]


        # # 4. Convert to PyTorch tensors
        X_train_tensor, y_train_tensor = generate_basis_tensor(X_train, y_train_fold, parameters)
        y_train_tensor = y_train_fold.clone().detach().float()

        X_test_tensor, y_test_tensor = generate_basis_tensor(X_test, y_test_fold, parameters)


        n_basis = X_train_tensor.shape[1] // n_features
        # print(X_train_tensor.shape, n_features, basic)
        # exit()
        # fitting the model here
        models = []
        for i, quantile in enumerate(parameters.quantile_tau_list):
            parameters.quantile_tau = quantile
            # print(X_train_tensor.shape)
            # exit()
            model = fit(X_train_tensor, y_train_tensor, n_features, n_basis, loss_fn, penalty_fn, parameters)
            models.append(model)
            y_pred = model(X_test_tensor)
            if quantile == .5:
                mse_losses.append(mse_loss(y_pred, y_test_tensor, parameters).item())
                mae_losses.append(mae_loss(y_pred, y_test_tensor, parameters).item())
            mcpe_losses[i] += quantile_loss(y_pred, y_test_tensor, parameters).item()
            qb_losses[quantile].append(quantile_bias_loss(y_pred, y_test_tensor, parameters).item())
            # print(X_train_tensor.shape)
            # print(y_train_tensor.shape)
            # plot_fit(model, X_full, y_full, formula)
        return models
def report(model, parameters: Parameters):
    global qb_losses, mse_loss, mae_losses, mcpe_losses, quantiles, mse_losses
    # print("----------------------------")
    

    per_quantile_qb_avg = [sum(qb_losses[q])/len(qb_losses[q]) for q in quantiles]
    print("----------------------------")
    print("lengths", len(mse_losses), len(mae_losses), len(mcpe_losses))
    print("Avg mse", sum(mse_losses)/len(mse_losses))
    print("Avg mae", sum(mae_losses)/len(mae_losses))
    print("Avg mcpe", sum(mcpe_losses)/len(mse_losses))
    print("Avg qb", sum(per_quantile_qb_avg)/len(per_quantile_qb_avg))
    print("min mse", min(mse_losses))
    print("min mae", min(mae_losses))
    print(per_quantile_qb_avg)
    qb_losses = defaultdict(list)
    mse_losses, mae_losses, mcpe_losses = [], [], [0 for _ in quantiles]
    for w in model.weights:
        print(w)
    if parameters.splam:
        print(type(model))
        print("linear weights: ", model.linear_weights)
    # if parameters.splam:
    #     print(model.linear_weights)
    # print(model.weights)
    # print(f"\nAverage Validation Loss over {k} folds: {avg_val_loss:.4f}")
    # return avg_val_loss

def g1(x): return .2*np.sin(x)
quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
mcpe_losses = [0 for _ in quantiles] # list of losses per quantile, list is length of folds

# quantiles = [.2 ,.5, .8]
# quantiles =[ .5]
basic = False
plotQuantiles = False

if not basic:
    labels, features = read_from_csv(meat, "fat")
    labels = labels.reshape(-1, 1)
else:
    # 1. Generate synthetic data
    np.random.seed(0)
    n = 50
    x1 = np.random.uniform(-3, 3, n)
# additive nonlinear regression with basis splines of quartic degree

    y_1 = g1(x1) #+ np.random.normal(0, 0.3, size=n)
    y = y_1
    y = y + .08 * np.random.normal(0, 0.5, size=n)
    y_True = y_1
    y = y.reshape(-1, 1)
    features, labels = x1, y
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    features = features.view(n, 1)
# print(features)
# print(labels)
# print(features.shape)

spline_formula = "bs(x, df=6, degree=3, include_intercept=True)"
linear_formula = "x"
#lambda used to be .012 for cubic spline, and .025 for splam 
model_parameters = Parameters(quantile_tau_list=quantiles, lambda_=.025, scad_alpha=3.7, scad_lambda=1.5, splam_alpha=.3, formula=spline_formula)
splam_parameters = Parameters(quantile_tau_list=quantiles, lambda_=.2, scad_alpha=3.7, scad_lambda=1.5, splam_alpha=.5, formula=spline_formula, splam=True)
splam_parameters2 = Parameters(quantile_tau_list=quantiles, lambda_=.2, scad_alpha=3.7, scad_lambda=1.5, splam_alpha=.7, formula=spline_formula, splam=True)

linear_params = Parameters(quantile_tau_list=quantiles, lambda_=.025, scad_alpha=3.7, scad_lambda=1.5, splam_alpha=.3, formula=linear_formula)

for i in range(7,10):
    cubic_model = cross_validate(features, labels, group_scad_penalty, quantile_loss, parameters=model_parameters, random_state=i, k=10)
report(cubic_model[0], model_parameters)
for i in range(7,10):
    splam_model = cross_validate(features, labels, group_scad_penalty, quantile_loss, parameters=splam_parameters, random_state=i, k=10)
report(splam_model[0], splam_parameters)

for i in range(7,10):
    splam_model2 = cross_validate(features, labels, group_scad_penalty, quantile_loss, parameters=splam_parameters2, random_state=i, k=10)
report(splam_model2[0], splam_parameters2)
# for i in range(7,10):
#     linear_model = cross_validate(features, labels, scad_penalty, quantile_loss, parameters=linear_params, random_state=i, k=5)
# report()
color_list = ["tab:green", "black", "red", "green", "blue", "brown"]
if basic and plotQuantiles:
    features, inds = torch.sort(features, dim=0)
    labels = labels[inds.squeeze()]
    X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, model_parameters)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axs[0].scatter(features, labels, label='Training $g_1$')
    axs[0].set_title('Cubic Splines')
    
    for i, model in enumerate(cubic_model):

        model_pred = model(X_train_tensor).detach().numpy()
        axs[0].plot(features, model_pred, label=f'Estimated quantile {quantiles[i]}', color=color_list[i])
    # axs[0].legend()



    X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, splam_parameters)
    axs[1].scatter(features, labels, label='Training $g_1$')
    axs[1].set_title('Splam Splines alpha = .7')
    
    for i, model in enumerate(splam_model):

        model_pred = model(X_train_tensor).detach().numpy()
        axs[1].plot(features, model_pred, label=f'Estimated quantile {quantiles[i]}', color=color_list[i])
    axs[1].legend()



    axs[2].scatter(features, labels, label='Training $g_1$')
    axs[2].set_title('Splam Splines alpha = .9')
    
    for i, model in enumerate(splam_model2):

        model_pred = model(X_train_tensor).detach().numpy()
        axs[2].plot(features, model_pred, label=f'Estimated quantile {quantiles[i]}', color=color_list[i])
    # axs[2].legend()
    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    # plt.tight_layout()
plt.show()
plt.savefig("basic_splam2.png")



if basic and not plotQuantiles:
    features, inds = torch.sort(features, dim=0)
    labels = labels[inds.squeeze()]
    # print(features)
    X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, model_parameters)
    model_pred = cubic_model(X_train_tensor).detach().numpy()

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].scatter(features, labels, label='Train', linestyle='--')
    axs[0].plot(features, model_pred, label='Estimated', color='red')
    axs[0].set_title('Cubic Splines')
    axs[0].legend()



    X_train_tensor, y_train_tensor = generate_basis_tensor(features, labels, splam_parameters)
    splam_model_pred = splam_model(X_train_tensor).detach().numpy()

    axs[1].scatter(features, labels, label='Train', linestyle='--')
    axs[1].plot(features, splam_model_pred, label='Estimated', color='red')
    axs[1].set_title('Cubic splines with linear feature')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Avg mse 0.044464402832090855
# Avg mae 0.16067576706409453
# Avg mcpe 0.08033788353204727

# print("mse_losses: ", mse_losses)
# print("mae_losses: ", mae_losses)
# print("mcpe_losses: ", mcpe_losses)
# print("qb_losses: ", qb_losses)
