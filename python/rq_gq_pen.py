import torch
from utils import *


def rq_gq_pen(x, y, tau, lam=None, nlambda=100, eps=None, weights=None,        \
              penalty_factor=None, scalex=None,tau_penalty_factor=None,        \
              gmma=0.2,  max_iter=200, lam_discard=True, converge_eps=1e-4,    \
              beta0=None):
    if eps is None:
        eps = 0.01 if (x.shape[0] < x.shape[1]) else 0.001

    ntau = tau.numel()
    np = x.shape
    n = np[0] 
    p = np[1]
    nng = ntau.repeat(p)
    # These types of default assignments can be added just be added to the 
    # function header, but for now I will leave it to have continuity w/ R code 
    if penalty_factor is None:
        penalty_factor = 1
    
    if weights is None:
        weights = torch.ones(n)

    if tau_penalty_factor is None:
        tau_penalty_factor = torch.ones(ntau)

    ## some initial checks
    if(ntau < 3):
        stop("please provide at least three tau values!")
    
    if(min(apply(x,2,sd))==0):
        stop("At least one of the x predictors has a standard deviation of zero")
    
    if(not is_unique_tensor(tau)):
        stop("All entries of tau should be unique")
    
    if(is_unsorted_tensor(tau)):
        stop("Quantile values should be entered in ascending order")
    
    if(y.numel() != x.shape[0]):
        stop("length of y and number of rows in x are not the same")

    if(weights is not None):
        if(weights.numel() != y.numel()):
            stop("number of weights does not match number of responses")
        if(torch.any(weights <= 0)):
            stop("all weights most be positive")
        
    # if(is.matrix(y)==TRUE):
    #     y <- as.numeric(y)
    
    # if(min(penalty.factor) < 0 | min(tau.penalty.factor) < 0):
    #     stop("Penalty factors must be non-negative.")

    # if(sum(penalty.factor)==0 | sum(tau.penalty.factor)==0):
    #     stop("Cannot have zero for all entries of penalty factors. This would be an unpenalized model")
    
    return