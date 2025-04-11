
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
# Activate automatic conversion between numpy and R
from rpy2.robjects import r
from scipy.stats import t
from rpy2.robjects import NULL
from rpy2.robjects import FloatVector
numpy2ri.activate()
# base = importr('base')
# ro.r('library(devtools)')
# # ro.r('install.packages("rqPen", repos = "http://cran.us.r-project.org")')

# ro.r('install.packages("/Users/aryanjoshi/Documents/EECS_classes/553-Project/rqpen", repos=NULL, dependencies=TRUE, type="source")')
# ro.r()

ro.r("""
    library(devtools)
    install.packages("/Users/aryanjoshi/Documents/EECS_classes/553-Project/rqpen", repos=NULL, type="source")
    library(rqPen)
    library(quantreg)
""")

def hrq_glasso_replacement(x, y, groups, subtau=.5, lambda_=NULL, penf=NULL, scalex=True, gamma=.2, max_iter=200, converge_eps=10**-4, lambda_discard=True, weights=NULL):
    ro.globalenv['x'] = x
    ro.globalenv['y'] = FloatVector(y)
    ro.globalenv['groups_mine'] = FloatVector(groups)
    ro.globalenv['tau'] = subtau
    ro.globalenv['lambda'] = lambda_
    ro.globalenv['penf'] = penf
    ro.globalenv['scalex'] = scalex
    ro.globalenv['gamma'] = gamma
    ro.globalenv['max_iter'] = max_iter
    ro.globalenv['converge_eps'] = converge_eps
    ro.globalenv['lambda_discard'] = lambda_discard
    ro.globalenv['weights'] = weights

    ro.r("""

        fit<- rq.hrq_glasso_helper(x, y, groups_mine, tau, lambda, penf, scalex, gamma, max_iter, converge_eps, lambda_discard, weights)
        rval <- fit
         
         """)
    
    return ro.globalenv['rval']


def check(x,tau=.5):
    return x*(tau - (x<0))

def scad(x, lambda_=1, a=3.7):
    absx = np.abs(x)
    pen = np.where(
        absx < lambda_,
        lambda_ * absx,
        np.where(
            absx < a * lambda_,
            ((a**2 - 1) * lambda_**2 - (absx - a * lambda_)**2) / (2 * (a - 1)),
            (a + 1) * lambda_**2 / 2
        )
    )
    return pen

def getPenfunc(penalty):
    penalty = penalty.upper()  # Normalize case
    if penalty in ["LASSO", "GLASSO", "ALASSO", "GADLASSO"]:
        return lasso
    elif penalty in ["SCAD", "GSCAD"]:
        return scad
    elif penalty in ["MCP", "GMCP"]:
        pass
    elif penalty == "ENET":
        pass
    elif penalty == "GQ":
        pass
    else:
        raise ValueError(f"Unsupported penalty type: {penalty}")
    
def rq_pen_modelreturn(coefs, x, y, tau, lambda_, local_penalty_factor, penalty, a, weights=None):
    penfunc = getPenfunc(penalty)
    return_val = {}

    # Ensure coefs is at least 2D for unified handling
    coefs = np.atleast_2d(coefs)
    is_vector = coefs.shape[0] == 1 or coefs.shape[1] == 1
    if is_vector:
        coefs = coefs.reshape(-1, 1)

    n = len(y)
    if weights is None:
        weights = np.ones(n)

    return_val["coefficients"] = coefs.copy()

    p = x.shape[1]
    x_names = [f"x{i+1}" for i in range(p)]
    x_names = ["intercept"] + x_names

    # Compute fitted values and residuals
    intercept = np.ones((x.shape[0], 1))
    design_matrix = np.hstack((intercept, x))  # shape: (n, p+1)
    fits = design_matrix @ coefs               # shape: (n, m)
    res = y.reshape(-1, 1) - fits              # shape: (n, m)

    if coefs.shape[1] == 1:
        # Single quantile case
        rho = np.mean(check(res.flatten(), tau) * weights)
        penalty_term = np.sum(penfunc(coefs[1:, 0], lambda_ * local_penalty_factor, a))
        return_val["rho"] = rho
        return_val["PenRho"] = rho + penalty_term
        return_val["nzero"] = np.sum(coefs != 0)
    else:
        # Multiple quantiles case
        rho = np.mean(check(res, tau) * weights.reshape(-1, 1), axis=0)
        return_val["rho"] = rho
        pen_rho = []
        nzero = []

        for i in range(coefs.shape[1]):
            pen = np.sum(penfunc(coefs[1:, i], lambda_[i] * local_penalty_factor, a))
            pen_rho.append(rho[i] + pen)
            nzero.append(np.sum(coefs[:, i] != 0))

        return_val["PenRho"] = np.array(pen_rho)
        return_val["nzero"] = np.array(nzero)

    return_val["tau"] = tau
    return_val["a"] = a
    return return_val

def rq_glasso (x, y, tau, groups, lambda_, group_pen_factor, scalex, tau_penalty_factor, max_iter, converge_eps, gamma, lambda_discard, weights):
    dims = x.shape
    n = dims[0]
    p = dims[1]
    g = len(set(groups))
    nt = len(tau)
    models = [None for _ in range(nt)]

    for i in range(nt):
        subtau = tau[i]
        penf = group_pen_factor*tau_penalty_factor[i]
        models[i] = hrq_glasso_replacement(x, y, groups, subtau, lambda_, penf, scalex, gamma, max_iter, converge_eps, lambda_discard, weights)
        models[i] = rq_pen_modelreturn(model[i]["beta"], x, y, subtau, models[i]["lambda"], np.ones(p), "gLASSO", 1, weights)
# ro.r("""
#      library(devtools)
# install.packages("/Users/aryanjoshi/Documents/EECS_classes/553-Project/rqpen", repos=NULL, type="source")
# library(rqPen)
# library(quantreg)
# set.seed(1)

# x <- matrix(rnorm(20*8,sd=1),ncol=8)
# y <- 1 + x[,1] + 3*x[,3] - x[,8] + rt(20,3)
# print(x)
# print(y)
# g <- c(1,1,1,2,2,2,3,3)
# tvals <- c(.25,.75)
# r1 <- rq.group.pen(x,y,groups=g,norm=2,penalty="gSCAD")



#      """)
n = 100
p = 10
x0 = np.random.randn(n, p)

# Create the X matrix by combining x0, x0^2, and x0^3
X = np.hstack([x0, x0**2, x0**3])

# Reorder columns as specified in the R code (order(rep(1:p,3)))
X = X[:, np.argsort(np.tile(np.arange(1, p+1), 3))]

# Calculate the response variable y
y = -2 + X[:, 0] + 0.5 * X[:, 1] - X[:, 2] - 0.5 * X[:, 6] + X[:, 7] - 0.2 * X[:, 8] + t(df=2).rvs(n)

X = np.array([
    [-0.62645381,  0.91897737, -0.1645236 ,  2.40161776, -0.5686687 , -0.62036668],
    [ 0.18364332,  0.7821363 , -0.2533617 , -0.039240003, -0.1351786 ,  0.04211587],
    [-0.83562861,  0.07456498,  0.6969634 ,  0.68973936,  1.178087  , -0.91092165],
    [ 1.5952808 , -1.9893517 ,  0.5566632 ,  0.02800216, -1.5235668 ,  0.15802877],
    [ 0.32950777,  0.61982575, -0.6887557 , -0.74327321,  0.5939462 , -0.65458464],
    [-0.82046838, -0.05612874, -0.7074952 ,  0.1887923 ,  0.3329504 ,  1.76728727],
    [ 0.48742905, -0.15579551,  0.364582  , -1.80495863,  1.0630998 ,  0.71670748],
    [ 0.73832471, -1.47075238,  0.7685329 ,  1.46555486, -0.3041839 ,  0.91017423],
    [ 0.57578135, -0.47815006, -0.1123462 ,  0.15325334,  0.3700188 ,  0.38418536],
    [-0.30538839,  0.41794156,  0.8811077 ,  2.17261167,  0.2670988 ,  1.68217608],
    [ 1.51178117,  1.35867955,  0.3981059 ,  0.47550953, -0.54252   , -0.63573645],
    [ 0.38984324, -0.10278773, -0.6120264 , -0.70994643,  1.2078678 , -0.46164473],
    [-0.62124058,  0.38767161,  0.3411197 ,  0.61072635,  1.1604026 ,  1.43228224],
    [-2.21469989, -0.05380504, -1.1293631 , -0.93409763,  0.7002136 , -0.65069635],
    [ 1.12493092, -1.37705956,  1.4330237 , -1.2536334 ,  1.5868335 , -0.20738074],
    [-0.04493361, -0.41499456,  1.9803999 ,  0.29144624,  0.5584864 , -0.39280793],
    [-0.01619026, -0.39428995, -0.3672215 , -0.44329187, -1.2765922 , -0.31999287],
    [ 0.94383621, -0.0593134 , -1.0441346 ,  0.00110535, -0.5732654 , -0.2791133 ],
    [ 0.8212212 ,  1.10002537,  0.5697196 ,  0.07434132, -1.2246126 ,  0.49418833],
    [ 0.59390132,  0.76317575, -0.1350546 , -0.58952095, -0.4734006 , -0.17733048]
])

# The y vector (response variable)
y = np.array([
    2.3855162,  0.7277089,  5.1192276,  2.8483932,  4.6860812, -0.6099700,
    0.8109169,  4.9755190,  2.4222970,  3.9744380,  2.0114657,  3.3982980,
    -1.3881513, -3.6465959,  7.5240459,  9.1248881, -1.4029189, -0.4704082,
    3.7914830,  1.5205284
])
# Create the group vector (repeat each integer 1:p, each 3 times)
group = np.tile(np.arange(1, p+1), 3)
group = np.array([1,1,2,2,3,3])
print(X.shape, y.shape)
print(group.shape)
fit = hrq_glasso_replacement(X, y, group)
print(fit)