# %%
import torch
import torch.nn as nn
import torch.optim as optim
!pip install torch-kde
from torchkde import KernelDensity

# %%
# Generate synthetic data
torch.manual_seed(42)
feature1 = torch.linspace(-1, 1, 100).reshape(-1, 1)
y_train = 2*feature1 + 1 + .15 * torch.randn_like(feature1)
x_train = feature1
# x_train = torch.cat((feature1, feature1**2, feature1**3), dim=1)
num_features = len(x_train[0])
# x_train
# x_train = 2 * feature1 + 1 + 0.5 * torch.randn_like(feature1)  # y = 2x + 1 + noise


# %%

import numpy as np
# Define Ridge Regression Model
class RidgeRegression(nn.Module):
    def __init__(self):
        super(RidgeRegression, self).__init__()
        self.w = nn.Parameter(torch.randn(num_features))  # Trainable weight
        self.b = nn.Parameter(torch.randn(1))  # Trainable bias

    def forward(self, x):
        return self.w * x + self.b

# Ridge loss function (MSE + L2 penalty)
def ridge_loss(y_pred, y_true, model, alpha=0.1):
    mse_loss = torch.mean((y_pred - y_true) ** 2)
    l2_penalty = alpha * (model.w ** 2).sum()  # L2 regularization on weights
    return mse_loss + l2_penalty

def ridge_loss_ratio(y_pred, y_true, model, alpha=0.1):
    mse_loss = torch.mean((y_pred - y_true) ** 2)
    l2_penalty = alpha * (densities * model.w ** 2).sum()  # L2 regularization on weights
    return mse_loss + l2_penalty

def l2_penalty(model, alpha):
    return alpha * (model.w ** 2).sum()  # L2 regularization on weights
def l2_weighted_penalty(model, alpha):
    # np.tensordot(alpha, )
    return alpha @ (model.w ** 2)

def quantile_loss(y_pred, y_true, tau=0.5):
    error = y_true - y_pred
    return torch.mean(torch.maximum(tau * error, (tau - 1) * error))


def huber_loss(y_pred, y_true, delta, model, alpha):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    penalty = l2_weighted_penalty(model, alpha)
    # print(penalty, torch.mean(torch.where(is_small_error, squared_loss, linear_loss)))
    return torch.mean(torch.where(is_small_error, squared_loss, linear_loss)) + (penalty)

def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part


# %%
from sklearn.neighbors import KernelDensity as sk_kd
import numpy as np

def get_densities(x_train):
    # X = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = x_train
    kde = sk_kd(kernel='gaussian', bandwidth=0.2).fit(X)
    kde_vals = kde.score_samples(X)
    exp_vals = np.exp(kde_vals)
    kde_sum = np.sum(np.exp(kde_vals))
    normalized =  exp_vals/kde_sum
    return normalized

# %%

# Initialize model and optimizer
model = RidgeRegression()
optimizer = optim.Adam(model.parameters(), lr=0.1)
densities = get_densities(x_train)
# Training loop
def fit(quantile=.1, huber_delta = .1):
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        # loss = huber_loss(y_pred, y_train, huber_delta, model, torch.tensor([.01, .1, 100]))1, 100]))
        # loss = ridge_loss(y_pred, y_train, model, 0, densities)
        loss = ridge_loss_ratio(y_pred, y_train, model, 0, densities)
        weighted_loss = (1/2 * loss).mean()
        weighted_loss.backward()
        # ward()
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Weighted loss: {weighted_loss.item()}")

    # Prirs
    print(f"Learned parameters: w = {str(model.w)}, b = {model.b.item()}")


# %%
fit()

# %%
multivariate_normal = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2))
# X = multivariate_normal.sample((1000,)) # create data
X = torch.tensor([[-1., -1.], [-1., -1.], [-1., -1.], [-1., -1.], [-1., -1.], [-2., -1.], [-3., -2.], [1., 1.], [2., 1.], [3., 2.]], requires_grad=True)
kde = KernelDensity(bandwidth=0.2, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

# X_new = multivariate_normal.sample((100,)) # create new data 
# logprob = kde.score_samples(X_new)
logprob = kde.score_samples(X)
logprob.grad_fn # is not None
print(logprob)
sm = torch.nn.Softmax(dim=0)
print(torch.exp(logprob))

# %%
def rq_glasso(x, y, tau, groups, lambda_, group_pen_factor, scalex, tau_penalty_factor, max_iter, converge_eps, gamma, lambda_discard,weights):
    p = x.shape[1]
    n = x.shape[0]
    nt = len(tau)
    
    models = [None for _ in range(nt)]
    for i in range(nt):
        tau_i = tau[i]
        penf = group_pen_factor * tau_penalty_factor[i]
        models[i] = 
        


# %%
from sklearn.neighbors import KernelDensity as sk_kd
import numpy as np
X = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = sk_kd(kernel='gaussian', bandwidth=0.2).fit(X)
kde_vals = kde.score_samples(X)
exp_vals = np.exp(kde_vals)
kde_sum = np.sum(np.exp(kde_vals))
normalized =  exp_vals/kde_sum
print(normalized)


# %%
import matplotlib.plt
plt.plot()


