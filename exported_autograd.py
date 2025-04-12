import torch
import torch.nn as nn
import torch.optim as optim
# !pip install torch-kde
from torchkde import KernelDensity
import matplotlib.pyplot as plt
# import numpy as np


# Generate synthetic data
torch.manual_seed(42)
# features vlaues from 1 to 100
feature1 = torch.linspace(-20, 20, 100).reshape(-1, 1)
x_train = torch.cat((feature1, feature1**2, feature1**3), dim=1)
y_test = 0*feature1

# (-1)*feature1**3 + 1*feature1**2+ + 6*feature1 -20
y_train = y_test + .5 * torch.randn_like(feature1)
plt.ylim(-4, 4)
plt.plot(feature1, y_test, color="orange",label="true")
# y_train = 10*feature1 + 1 + 0 * torch.randn_like(feature1)

y_train = y_train.squeeze()

# x_train = feature1

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
  def __init__(self,lambda_=None, scad_alpha=None, scad_lambda=None, splam_alpha=None, tau=None, huber_alpha=None, huber_delta=None):
        self.lambda_ = lambda_
        
        self.scad_alpha = scad_alpha
        self.scad_lambda = scad_lambda
        self.splam_alpha = splam_alpha
        self.quantile_tau = tau
        self.huber_alpha = huber_alpha
        self.huber_delta = huber_delta
    
# Ridge loss function (MSE + L2 penalty)
def chosen_loss(y_pred, y_true, model, penalty, parameters):
    mse_loss = torch.mean((y_pred - y_true) ** 2)
    l2_penalty = penalty(model, parameters)
    # l2_penalty = penalty(model, scad_lambda, alpha_)  # L2 regularization on weights
    # print(type(l2_penalty))
    return mse_loss + parameters.lambda_ * l2_penalty

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
    epochs = 30000
    for epoch in range(epochs):
        optimizer.zero_grad()
        # y_pred = model(x_train)
        y_pred = x_train@model.w + model.b

        # loss = huber_loss(y_pred, y_train, huber_delta, model, torch.tensor([.01, .1, 100]))1, 100]))
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 1, alpha_ = 0.1)
        # loss = quantile_loss(y_pred, y_train, quantile)
        p = Parameters(lambda_=1.8)
        loss = chosen_loss(y_pred, y_train, model, l2_penalty, p)
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
# print(model_50.w)
# print(model_10.w)
# print(preds)
  # 100 points from -10 to 10
# feature1 = torch.linspace(-1, 1, 5).reshape(-1, 1)
# x_pltensor.detach().cpu().numpy()

# x_line = torch.cat((x, x**2, x**3), dim=1)
# # x_line = torch.cat((x.unsqueeze(1), (x**2).unsqueeze(1), (x**3).unsqueeze(1)), dim=1)
# y = model.w.detach() * x_line + model.b.item()

plt.scatter(feature1, y_train)
x = feature1
def plot_model(model, color):
    y_pred = x_train@model.w + model.b
    y = y_pred.detach().numpy()
    plt.plot(feature1, y , label="preds", color=color)

# y_pred_90 = x_train@model_90.w + model_90.b
# y_pred_10 = x_train@model_10.w + model_10.b
# y_pred_50 = x_train@model_50.w + model_50.b
# y = x_train@model.w + model.b

plot_model(model_90, "r")
# plot_model(model_50, "b")
# plot_model(model_10, "g")



plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# plt.plot(y, label="pred")
plt.show()


# multivariate_normal = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2))
# # X = multivariate_normal.sample((1000,)) # create data
# X = torch.tensor([[-1., -1.], [-1., -1.], [-1., -1.], [-1., -1.], [-1., -1.], [-2., -1.], [-3., -2.], [1., 1.], [2., 1.], [3., 2.]], requires_grad=True)
# kde = KernelDensity(bandwidth=0.2, kernel='gaussian') # create kde object with isotropic bandwidth matrix
# _ = kde.fit(X) # fit kde to data

# X_new = multivariate_normal.sample((100,)) # create new data 
# logprob = kde.score_samples(X_new)
# logprob = kde.score_samples(X)
# logprob.grad_fn # is not None
# print(logprob)
# sm = torch.nn.Softmax(dim=0)
# print(torch.exp(logprob))



# def rq_glasso(x, y, tau, groups, lambda_, group_pen_factor, scalex, tau_penalty_factor, max_iter, converge_eps, gamma, lambda_discard,weights):
#     p = x.shape[1]
#     n = x.shape[0]
#     nt = len(tau)
    
#     models = [None for _ in range(nt)]
#     for i in range(nt):
#         tau_i = tau[i]
#         penf = group_pen_factor * tau_penalty_factor[i]
#         models[i] = 
        


# from sklearn.neighbors import KernelDensity as sk_kd
# import numpy as np
# X = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# kde = sk_kd(kernel='gaussian', bandwidth=0.2).fit(X)
# kde_vals = kde.score_samples(X)
# exp_vals = np.exp(kde_vals)
# kde_sum = np.sum(np.exp(kde_vals))
# normalized =  exp_vals/kde_sum
# print(normalized)
