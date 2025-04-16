import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

from rpy2.robjects import NULL
from rpy2.robjects import FloatVector

def stop(err):
    raise ValueError(err)


def is_unique_tensor(t):
    """
    Returns true if all elements of input tensor are unique
    Params:
    t (torch.tensor): tensor to check for unique values
    """
    return t.numel() == torch.unique(t).numel()


def is_unsorted_tensor(t):
    """
    Return true if tensor is not sorted, false if it is
    Params:
    t (torch.tensor): tensor to check for sortedness
    """
    if t.numel() == 0:
        return False
    val = t[0]
    for num in t:
        if num < val:
            return True
        val = num
    return False


def tensor_IQR(t, q1=0.25, q2=0.75):
    """
    Returns the interquartile range (IQR) of t.
    You can specify a custom quantile range by passing values to q1, q2
    Params:
    t (torch.tensor): tensor to calculate IQR on
    q1 (optional, float, default 0.25): lower quantile. takes value in [0, 1]
    q2 (optional, float, default 0.75): upper quantile. takes value in [0, 1]
    """
    if q1 >= q2:
        return 0.0
    quant1 = torch.quantile(t, q1)
    quant2 = torch.quantile(t, q2)
    return quant2 - quant1


def scale(y, c=True, sc=True):
    """
    (I think) A Python port of the R function "scale." Found this on SO
    https://stackoverflow.com/questions/18005305/implementing-r-scale-function-in-pandas-in-python
    """
    # x = y.copy() I don't think we need to make a copy for our purposes

    if c:
        y -= y.mean()
    if sc and c:
        y /= y.std()
    elif sc:
        y /= torch.sqrt(y.pow(2).sum().div(y.count() - 1))
    return y


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


class R_ops():
    def __init__(self, rqpen_path="~/code/EECS553/553-Project/rqpen", recompile=False):
        self.rqpen_path = rqpen_path
        numpy2ri.activate()
        if recompile:
            ro.r(f"""
                library(devtools)
                install.packages("{self.rqpen_path}", repos=NULL, type="source")
                library(rqPen)
                library(quantreg)
            """)
        else:
            ro.r(f"""
                library(devtools)
                library(rqPen)
                library(quantreg)
            """)
    

    def hrq_glasso(self, x, y, groups, subtau=.5, lambda_=NULL, penf=NULL, scalex=True, gamma=.2, max_iter=200, converge_eps=10**-4, lambda_discard=True, weights=NULL):
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
    
    #getLamMaxGroup <- function(x,y,group.index,tau=.5,group.pen.factor,gamma=.2,gamma.max=4,gamma.q=.1,penalty="gLASSO",scalex=TRUE,tau.penalty.factor,norm=2,weights=NULL)
    def getLamMaxGroup(
        self,
        x=None,
        y=None,
        group_index=None,
        tau=0.5,
        group_pen_factor=None,
        gamma=0.2,
        gamma_max=4,
        gamma_q=0.1,
        penalty="gLASSO",
        scalex=True,
        tau_penalty_factor=None,
        norm=2,
        weights=None,
    ):
        '''
        This function assumes that all array inputs are torch tensors
        '''
        if (
            (x is None)
            | (y is None)
            | (group_index is None)
            | (group_pen_factor is None)
            | (tau_penalty_factor is None)
        ):
            stop("getLamMaxGroup missing parameter")
        
        ro.globalenv['x'] = x.numpy()
        ro.globalenv['y'] = FloatVector(y.numpy())
        ro.globalenv['groups.index'] = FloatVector(group_index.numpy())
        ro.globalenv['tau'] = tau.numpy()
        ro.globalenv['group.pen.factor'] = group_pen_factor.numpy()
        ro.globalenv['gamma'] = gamma
        ro.globalenv['gamma.max'] = gamma_max
        ro.globalenv['gamma.q'] = gamma_q
        ro.globalenv['penalty'] = penalty
        ro.globalenv['scalex'] = scalex
        ro.globalenv['tau.penalty.factor'] = tau_penalty_factor.numpy()
        ro.globalenv['norm'] = norm
        ro.globalenv['weights'] = weights.numpy() if weights is not None else NULL
        is_there = ro.r('exists("getLamMaxGroup", where = asNamespace("rqPen"), inherits = FALSE)')
        out = ro.r("""

            lam <- rqPen:::getLamMaxGroup(x, y, groups.index, tau, group.pen.factor, gamma, gamma.max, gamma.q, penalty, scalex, tau.penalty.factor, norm, weights)
            rval <- lam
            
            """)
        return ro.globalenv['rval']
    
    
    def rq_group_pen(self, x, y, penalty="gLASSO"):
        '''
        Just let everything default for testing, add parameters if you really need to test something
        specific. The inputs are assumed to be torch tensors
        '''
        ro.globalenv['x'] = x.numpy()
        ro.globalenv['y'] = FloatVector(y.numpy())
        ro.globalenv['penalty'] = penalty
        out = ro.r("""

            lam <- rqPen:::rq.group.pen(x, y)
            rval <- lam
            
            """)
        return ro.globalenv['rval']
    

    def rq_glasso(self, x, y, tau, groups, lambda_, group_pen_factor, scalex, tau_penalty_factor, max_iter, converge_eps, gamma, lambda_discard, weights):
        dims = x.shape
        n = dims[0]
        p = dims[1]
        g = len(set(groups))
        nt = len(tau)
        models = [None for _ in range(nt)]

        if weights is None:
            weights = np.ones(len(y))
        for i in range(nt):
            subtau = tau[i]
            penf = group_pen_factor*tau_penalty_factor[i]
            models[i] = self.hrq_glasso(x, y, groups, subtau, lambda_, penf, scalex, gamma, max_iter, converge_eps, lambda_discard, weights)
            models[i] = rq_pen_modelreturn(models[i]["beta"], x, y, subtau, models[i]["lambda"], np.ones(p), "gLASSO", 1, weights)


    def rq_finish_group_pen(
        self,
        x,
        y,
        tau,
        groups,
        penalty,
        lamb,
        nlambda,
        eps,
        alg,
        a,
        norm,
        group_pen_factor,
        tau_penalty_factor,
        scalex,
        coef_cutoff,
        max_iter,
        converge_eps,
        gamma,
        lambda_discard,
        weights,
        penalty_factor,
    ):
        ro.globalenv['x'] = x.numpy()
        ro.globalenv['y'] = FloatVector(y.numpy())
        ro.globalenv['group.index'] = FloatVector(groups.numpy() + 1)
        ro.globalenv['tau'] = tau.numpy()
        ro.globalenv['group.pen.factor'] = group_pen_factor.numpy()
        ro.globalenv['gamma'] = gamma
        ro.globalenv['penalty'] = penalty
        ro.globalenv['lambda'] = lamb.numpy()
        ro.globalenv['nlambda'] = nlambda
        ro.globalenv['eps'] = eps
        ro.globalenv['alg'] = alg
        ro.globalenv['a'] = a.item()
        ro.globalenv['scalex'] = scalex
        ro.globalenv['coef.cutoff'] = coef_cutoff
        ro.globalenv['max.iter'] = max_iter
        ro.globalenv['tau.penalty.factor'] = tau_penalty_factor.numpy()
        ro.globalenv['norm'] = norm
        ro.globalenv['converge.eps'] = converge_eps
        ro.globalenv['lambda.discard'] = lambda_discard
        ro.globalenv['penalty.factor'] = penalty_factor.numpy()
        print(weights)
        ro.globalenv['weights'] = weights.numpy() if weights is not None else NULL

        out = ro.r("""

            lam <- rqPen:::rq.finish.group.pen(x, y, tau, group.index, penalty, lambda, nlambda, eps, alg, a, norm, group.pen.factor, tau.penalty.factor, scalex, coef.cutoff, max.iter, converge.eps, gamma, lambda.discard, weights, penalty.factor)
            rval <- lam
            
            """)
        
        return ro.globalenv['rval']