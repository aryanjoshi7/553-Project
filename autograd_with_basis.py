import torch
import torch.nn as nn
import torch.optim as optim
# !pip install torch-kde
from torchkde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix

# Generate synthetic data
torch.manual_seed(42)
# features vlaues from 1 to 100
# feature1 = torch.linspace(-20, 20, 100).reshape(-1, 1)

# x_train = torch.cat((feature1, feature1**2, feature1**3), dim=1)
# y_test = 4*feature1

# # (-1)*feature1**3 + 1*feature1**2+ + 6*feature1 -20
# y_train = y_test + .5 * torch.randn_like(feature1)
# # plt.ylim(-4, 4)
# plt.plot(feature1, y_test, color="orange",label="true")
# # y_train = 10*feature1 + 1 + 0 * torch.randn_like(feature1)

# y_train = y_train.squeeze()



n = 100
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

def f1(x): return np.sin(x)
def f2(x): return np.log1p(x)
y1_test = f1(x1)
y2_test = f2(x2)
y_test = f1(x1) + f2(x2)
y = f1(x1) + f2(x2) + np.random.normal(0, 0.3, size=n)

# Step 2: Create basis design matrix (B-splines for each feature)
X1_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x1})
X2_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x2})
X_np = np.hstack([X1_design, X2_design])
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
x_train = X_tensor
y_train = y_tensor

num_features = x_train.shape[1]
print("num_features", num_features)

# X = torch.tensor([[-1,-10],[-1,-10], [-1,-10], [-1,-10], [-1,-10], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# unweighted_X = torch.tensor([[-1, -1], [1, 1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# x_train = X[:, 0]
# y_train = X[:, 1]
# num_features = 1

# x_train
# x_train = 2 * feature1 + 1 + 0.5 * torch.randn_like(feature1)  # y = 2x + 1 + noise


# Define Ridge Regression Model
class RidgeRegression(nn.Module):
    def __init__(self):
        super(RidgeRegression, self).__init__()
        self.w = nn.Parameter(torch.zeros(num_features, requires_grad=True))  # Trainable weight
        self.b = nn.Parameter(torch.zeros(1, requires_grad=True))  # Trainable bias

    def forward(self, x):
        # for i in range(len)
        # x_train[0][0] * weights[0], x_train[0][1] * weights[1], x_train[0][2] * weights[2]
        preds = []
        for i in range(len(x_train)):
            pred = x[i][0] * self.w[0] + x[i][1] * self.w[1] + x[i][2] * self.w[2]
            preds.append(pred)
        return torch.FloatTensor(preds)
    
class Parameters:
  def __init__(self,lambda_=None, scad_alpha=None, scad_lambda=None, splam_alpha=None, quantile_tau=None, huber_alpha=None, huber_delta=None):
        self.lambda_ = lambda_
        
        self.scad_alpha = scad_alpha
        self.scad_lambda = scad_lambda
        self.splam_alpha = splam_alpha
        self.quantile_tau = quantile_tau
        self.huber_alpha = huber_alpha
        self.huber_delta = huber_delta
    
# Ridge loss function (MSE + L2 penalty)
def chosen_loss(y_pred, y_true, model, loss, penalty, parameters):
    loss_calc = loss(model, y_pred, y_true, parameters)
    # torch.mean((y_pred - y_true) ** 2)
    l2_penalty = penalty(model, parameters)
    # l2_penalty = penalty(model, scad_lambda, alpha_)  # L2 regularization on weights
    # print(type(l2_penalty))
    return loss_calc + parameters.lambda_ * l2_penalty

def ridge_loss_ratio_covariate_shift(y_pred, y_true, model, densities, alpha=0.1):
    densities = torch.tensor(densities, dtype=y_pred.dtype, device=y_pred.device)
    mse_loss = torch.mean(1/densities * (y_pred - y_true) ** 2)
    l2_penalty = alpha * (model.w ** 2).sum()  # L2 regularization on weights
    return mse_loss + l2_penalty

def splam_penalty(model, parameters: Parameters):
    penalty = parameters.splam_alpha*model.w[0]**2
    penalty += sum((1 - parameters.splam_alpha) * model.w[1:] ** 2)
    return penalty
    

def l2_penalty(model, parameters: Parameters):
    return (model.w ** 2).sum()  # L2 regularization on weights

def l2_weighted_penalty(model, parameters: Parameters):
    # np.tensordot(alpha, )
    return parameters.alpha @ (model.w ** 2)

def mse_loss(model, y_pred, y_true, p: Parameters):
    return torch.mean((y_pred - y_true) ** 2)

def quantile_loss(model, y_pred, y_true, parameters: Parameters):
    error = y_true - y_pred
    return torch.mean(torch.maximum(parameters.quantile_tau * error, (parameters.quantile_tau - 1) * error))


def huber_loss(model, parameters: Parameters):
    p = parameters
    error = p.y_true - p.y_pred
    is_small_error = torch.abs(error) <= p.huber_delta
    squared_loss = 0.5 * error**2
    linear_loss = p.huber_delta * (torch.abs(error) - 0.5 * p.huber_delta)
    penalty = l2_weighted_penalty(model, p.huber_alpha)
    # print(penalty, torch.mean(torch.where(is_small_error, squared_loss, linear_loss)))
    return torch.mean(torch.where(is_small_error, squared_loss, linear_loss)) + (penalty)

def scad_penalty(model, parameters: Parameters):
    p = parameters
    # beta_hat = model.w.detach().numpy()
    is_linear = (np.abs(model) <= p.scad_lambda)
    is_quadratic = np.logical_and(p.scad_lambda < np.abs(model), np.abs(model) <= p.scad_alpha * p.scad_lambda)
    is_constant = (p.scad_alpha * p.scad_lambda) < np.abs(model)
    
    linear_part = p.scad_lambda * np.abs(model) * is_linear
    quadratic_part = (2 * p.scad_alpha * p.scad_lambda * np.abs(model) - model**2 - p.scad_lambda**2) / (2 * (p.scad_alpha - 1)) * is_quadratic
    constant_part = (p.scad_lambda**2 * (p.scad_alpha + 1)) / 2 * is_constant
    # print(linear_part + quadratic_part + constant_part)
    return sum(linear_part + quadratic_part + constant_part)



from sklearn.neighbors import KernelDensity as sk_kd
import numpy as np

def get_densities(x_train):
    # X = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    x_train = x_train.reshape(-1, 1)
    # print("reshaped", x_train)
    kde = sk_kd(kernel='gaussian', bandwidth=0.2).fit(x_train)
    kde_vals = kde.score_samples(x_train)
    # print("kde_vals", kde_vals)
    
    exp_vals = np.exp(kde_vals)
    kde_sum = np.sum(np.exp(kde_vals))
    normalized =  exp_vals/kde_sum
    return normalized



# Initialize model and optimizer

densities = get_densities(x_train)
# Training loop
def fit(quantile=.1, huber_delta = .01, _lambda_ = 0):
    model = RidgeRegression()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        # y_pred = model(x_train)
        y_pred = x_train@model.w + model.b

        # loss = huber_loss(y_pred, y_train, huber_delta, model, torch.tensor([.01, .1, 100]))1, 100]))
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 1, alpha_ = 0.1)
        # loss = quantile_loss(y_pred, y_train, quantile)
        p = Parameters(lambda_=0, quantile_tau=.5)
        loss = chosen_loss(y_pred, y_train, model, mse_loss, l2_penalty, p)
        # loss = torch.mean((y_pred - y_train)**2)

        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 2, alpha_ = 0.05)

        # loss = ridge_loss_ratio(y_pred, y_train,model, densities, 0) # covariate shift
        # weighted_loss = (loss).mean()
        # weighted_loss
        # ward()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Prirs
    print(f"Learned parameters: w = {str(model.w)}, b = {model.b.item()}")
    return loss, model

error, model_90 = fit(quantile = .5, _lambda_ = 0.5)
# error, model_10 = fit(quantile = .1)
# error, model_50 = fit(quantile = .5)

print(error.item())
print(model_90.w)

weights = model_90.w.detach().numpy().flatten()

# Split weights by feature
w1 = weights[:7]  # weights for x1 spline basis
w2 = weights[7:]  # weights for x2 spline basis

# Construct basis again on a smooth grid for plotting
x1_grid = np.linspace(0, 10, 100)
x2_grid = np.linspace(0, 10, 100)

X1_grid_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x1_grid})
X2_grid_design = dmatrix("bs(x, df=6, degree=3, include_intercept=False)", {"x": x2_grid})

# f1(x1): sum of (basis * weights)
print("X1_grid_design", X1_grid_design.shape, "w1", w1.shape)
f1_hat = X1_grid_design @ w1
f2_hat = X2_grid_design @ w2

# Plot f1
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x1_grid, f1_hat, label="Estimated f1(x1)")
# print(f1_hat)
# plt.plot(x1_grid, f1(x1_grid))
plt.title("Estimated f1(x1)")
plt.xlabel("x1")
plt.ylabel("f1(x1)")
plt.grid(True)
plt.legend()

# Plot f2
plt.subplot(1, 2, 2)
plt.plot(x2_grid, f2_hat, label="Estimated f2(x2)", color="orange")
plt.title("Estimated f2(x2)")
# plt.plot(x2_grid, f2(x2_grid))

plt.xlabel("x2")
plt.ylabel("f2(x2)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()