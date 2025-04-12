import torch
import torch.nn as nn
import torch.optim as optim
# !pip install torch-kde
from torchkde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import inspect


# Generate synthetic data
torch.manual_seed(42)
# torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# features vlaues from 1 to 100
# feature1 = torch.linspace(-1, -30, 30).reshape(-1, 1)
# plt.plot(feature1)
feature1 = torch.linspace(-30, -1, 30).reshape(-1, 1)
y_train = 2*feature1 + 1 + .15 * torch.randn_like(feature1)
print(y_train)
x_train = feature1
# x_train = torch.cat((feature1, feature1**2, feature1**3), dim=1)
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
        self.w = nn.Parameter(torch.randn(num_features))  # Trainable weight
        self.b = nn.Parameter(torch.randn(1))  # Trainable bias

    def forward(self, x):
        # x should only be 1 datapoint

        return x @ self.w + self.b


def pred_and_get_error(model, x, y):
    y_pred = []
    err = 0
    for datapoint in x:
        y_pred.append(model(datapoint).item())
        err += (y - y_pred[-1])**2
    return y_pred, err
        
        

# Ridge loss function (MSE + L2 penalty)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

def quantile_loss(y_pred, y_true, tau=0.5):
    error = y_true - y_pred
    return torch.mean(torch.max(tau * error, (tau - 1) * error))

def huber_loss(y_pred, y_true, model, alpha=0.2, delta=0.2):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    penalty = weighted_l2_penalty(model, alpha)
    return torch.mean(torch.where(is_small_error, squared_loss, linear_loss)) + (penalty)



def l2_penalty(model):
    return (model.w ** 2).sum()

def weighted_l2_penalty(model, alpha):
    return alpha @ (model.w ** 2).sum()

def splam_penalty(model, splam_a = 0.5): # don't think this is right if we were to scale up, fine for 2D
    penalty = splam_a*model.w[0]**2
    penalty += sum((1 - splam_a) * model.w[1:] ** 2)
    return penalty

def scad_penalty(model, scad_lambda, scad_a):
    beta_hat = model.w.detach().numpy()
    is_linear = (np.abs(beta_hat) <= scad_lambda)
    is_quadratic = np.logical_and(scad_lambda < np.abs(beta_hat), np.abs(beta_hat) <= scad_a * scad_lambda)
    is_constant = (scad_a * scad_lambda) < np.abs(beta_hat)
    
    linear_part = scad_lambda * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * scad_a * scad_lambda * np.abs(beta_hat) - beta_hat**2 - scad_lambda**2) / (2 * (scad_a - 1)) * is_quadratic
    constant_part = (scad_lambda**2 * (scad_a + 1)) / 2 * is_constant
    return sum(linear_part + quadratic_part + constant_part)


def filter_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in valid_params}


def chosen_loss(y_pred, y_true, model, loss_func = mse_loss, regularization_func = l2_penalty, _lambda_ = 0, **additional_params):
    loss_kwargs = filter_kwargs(loss_func, **additional_params)
    reg_kwargs = filter_kwargs(regularization_func, **additional_params)

    loss = loss_func(y_pred, y_true, **loss_kwargs)
    regularization = regularization_func(model, **reg_kwargs)
    # print("_lambda_: ", _lambda_)
    return loss + _lambda_ * regularization


# def ridge_loss(y_pred, y_true, model, penalty, scad_lambda = 0.3, lambda_ = 0.1, alpha_ =0.1):
#     mse_loss = torch.mean((y_pred - y_true) ** 2)
#     l2_penalty = penalty(model, alpha_)
#     # l2_penalty = penalty(model, scad_lambda, alpha_)  # L2 regularization on weights
#     # print(type(l2_penalty))
#     return mse_loss + lambda_ * l2_penalty

# def ridge_loss_ratio(y_pred, y_true, model, densities, alpha=0.1):
#     densities = torch.tensor(densities, dtype=y_pred.dtype, device=y_pred.device)
#     mse_loss = torch.mean(1/densities * (y_pred - y_true) ** 2)
#     l2_penalty = alpha * (model.w ** 2).sum()  # L2 regularization on weights
#     return mse_loss + l2_penalty

    



# from sklearn.neighbors import KernelDensity as sk_kd
# import numpy as np

# def get_densities(x_train):
#     # X = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#     x_train = x_train.reshape(-1, 1)
#     # print("reshaped", x_train)
#     kde = sk_kd(kernel='gaussian', bandwidth=0.2).fit(x_train)
#     kde_vals = kde.score_samples(x_train)
#     # print("kde_vals", kde_vals)
    
#     exp_vals = np.exp(kde_vals)
#     kde_sum = np.sum(np.exp(kde_vals))
#     normalized =  exp_vals/kde_sum
#     return normalized
# densities = get_densities(x_train)



# Initialize model and optimizer
model = RidgeRegression()
optimizer = optim.Adam(model.parameters(), lr=0.1)


# Training loop
def fit(quantile=.1, huber_delta = .1):
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)

        loss = chosen_loss(y_pred, y_train, model, mse_loss) # classic MSE
        # loss = chosen_loss(y_pred, y_train, model, mse_loss, l2_penalty, _lambda_ = 0.5) # ridge regression
        # loss = chosen_loss(y_pred, y_train, model, mse_loss, scad_penalty, _lambda_ = 5, scad_lambda = 5, scad_a = 0.5 ) # mse with scad
        # loss = chosen_loss(y_pred, y_train, model, mse_loss, splam_penalty, _lambda_ = 0.5, splam_a = 0.5 ) # mse with splam
        
        # loss = chosen_loss(y_pred, y_train, model, mse_loss, l2_penalty, _lambda_ = 0.2)





        # loss = huber_loss(y_pred, y_train, huber_delta, model, torch.tensor([.01, .1, 100]))1, 100]))
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 1, alpha_ = 0.1)
        # loss = ridge_loss(y_pred, y_train, model, splam_penalty, scad_lambda = 3 , lambda_ = 0, alpha_ = 0.05)

        # loss = ridge_loss_ratio(y_pred, y_train,model, densities, 0) # covariate shift
        # weighted_loss = (loss).mean()
        # weighted_loss
        # ward()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Prirs
    print(f"Learned parameters: w = {str(model.w)}, b = {model.b.item()}")
    return mse_loss(y_pred, y_train)
    # return ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 0, alpha_ = 0.05)

accuracy = fit()
print(accuracy.item())



# x = torch.linspace(-50, 50, 100).reshape(-1,1)  # 100 points from -10 to 10
# y = 10*x
# feature1 = torch.linspace(-1, 1, 5).reshape(-1, 1)
# x_pltensor.detach().cpu().numpy()

# x_line = torch.cat((x, x**2, x**3), dim=1)
# x_line = torch.cat((x.unsqueeze(1), (x**2).unsqueeze(1), (x**3).unsqueeze(1)), dim=1)
# y = model.w.detach() * x_line + model.b.item()
# y = x + x + x + model.b.item()

y_pred,_ = pred_and_get_error(model,x_train, y_train)
# print("y_pred", y_pred)


print("feature1:", feature1.shape)
print("y_pred:", len(y_pred))
print("types of pred", type(y_pred))
print("type of element", type(y_pred[0]))
# exit()
plt.plot(feature1.detach().numpy(), np.array(y_pred), label="Graph")

# print("densities", densities)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of y = mx + b')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.scatter(feature1, y_train)
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
