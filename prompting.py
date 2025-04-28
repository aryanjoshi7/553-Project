import torch
import torch.nn as nn
import torch.optim as optim
# !pip install torch-kde
from torchkde import KernelDensity
import matplotlib.pyplot as plt


# Generate synthetic data
torch.manual_seed(42)
# features vlaues from 1 to 100
feature1 = torch.linspace(-20, 20, 4).reshape(-1, 1)
y_train = 10*feature1 + 1 + 0 * torch.randn_like(feature1)
x_train = torch.cat((feature1, feature1**2, feature1**3), dim=1)
y_train = y_train.squeeze()
# x_train = feature1

num_features = x_train.shape[1]



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

# Ridge loss function (MSE + L2 penalty)
def ridge_loss(y_pred, y_true, model, penalty, scad_lambda = 0.3, lambda_ = 0.1, alpha_ =0.1):
    mse_loss = torch.mean((y_pred - y_true) ** 2)
    l2_penalty = penalty(model, alpha_)
    return mse_loss + lambda_ * l2_penalty

def ridge_loss_ratio(y_pred, y_true, model, densities, alpha=0.1):
    densities = torch.tensor(densities, dtype=y_pred.dtype, device=y_pred.device)
    mse_loss = torch.mean(1/densities * (y_pred - y_true) ** 2)
    l2_penalty = alpha * (model.w ** 2).sum()  # L2 regularization on weights
    return mse_loss + l2_penalty

def splam_penalty(model, alpha):
    penalty = alpha*model.w[0]**2
    penalty += sum((1 - alpha) * model.w[1:] ** 2)
    return penalty
    

def l2_penalty(model, alpha):
    return (model.w ** 2).sum()
def l2_weighted_penalty(model, alpha):
    return alpha @ (model.w ** 2)


def quantile_loss(y_pred, y_true, tau=0.5):
    error = y_true - y_pred
    return torch.mean(torch.maximum(tau * error, (tau - 1) * error))





# Initialize model and optimizer
model = RidgeRegression()
optimizer = optim.Adam(model.parameters(), lr=0.001)
densities = get_densities(x_train)
# Training loop
def fit(quantile=.1, huber_delta = .01):
    epochs = 20000
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        # loss = huber_loss(y_pred, y_train, huber_delta, model, torch.tensor([.01, .1, 100]))1, 100]))
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, scad_penalty, scad_lambda = 3 , lambda_ = 3, alpha_ = 0.3)
        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 1, alpha_ = 0.1)
        loss = torch.mean((y_pred - y_train)**2)

        # loss = ridge_loss(y_pred, y_train, model, l2_penalty, scad_lambda = 3 , lambda_ = 2, alpha_ = 0.05)

        # loss = ridge_loss_ratio(y_pred, y_train,model, densities, 0) # covariate shift
        # weighted_loss = (loss).mean()
        # weighted_loss
        # ward()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Prirs
    print(f"Learned parameters: w = {str(model.w)}, b = {model.b.item()}")
    return ridge_loss(y_pred, y_train, model, splam_penalty, scad_lambda = 3 , lambda_ = 6, alpha_ = 0.05), model

error, model = fit()
print(error.item())
model.w.requires_grad = False
weights = model.w
b = model.b
print(b.item())
print(x_train[0])
print(x_train[0][0] * weights[0], x_train[0][1] * weights[1], x_train[0][2] * weights[2])
preds = []
for i in range(len(x_train)):
    pred = x_train[i][0] * weights[0] + x_train[i][1] * weights[1] + x_train[i][2] * weights[2]
    preds.append(pred)
print(preds)
  # 100 points from -10 to 10
# feature1 = torch.linspace(-1, 1, 5).reshape(-1, 1)
# x_pltensor.detach().cpu().numpy()

# x_line = torch.cat((x, x**2, x**3), dim=1)
# # x_line = torch.cat((x.unsqueeze(1), (x**2).unsqueeze(1), (x**3).unsqueeze(1)), dim=1)
# y = model.w.detach() * x_line + model.b.item()
x = feature1
y = preds
plt.plot(x, y, label="Graph")
print("densities", densities)
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
