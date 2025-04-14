import torch
from utils import *
from workHorse import getA, getLamMaxGroup
import numpy as np


def rq_group_pen(
    x,
    y,
    tau=torch.tensor([0.5]),
    groups=None,
    penalty="gLASSO",
    lamb=None,
    nlambda=100,
    eps=None,
    alg="huber",
    a=torch.tensor([1]),
    norm=2,
    group_pen_factor=None,
    tau_penalty_factor=None,
    scalex=True,
    coef_cutoff=1e-8,
    max_iter=5000,
    converge_eps=1e-4,
    gamma=None,
    lambda_discard=True,
    weights=None,
    R=None,
    *args
):
    """
    Params:
    x (torch.tensor):
    y (torch.tensor):
    tau (float):
    """
    if eps is None:
        eps = 0.05 if (x.shape[0] < x.shape[1]) else 0.01

    tau_penalty_factor = (
        torch.ones(tau.numel()) if (tau_penalty_factor is None) else tau_penalty_factor
    )

    gamma = tensor_IQR(y) / 10 if (gamma is None) else gamma

    groups = torch.arange(0, x.shape[1]).int() if (groups is None) else groups

    if not is_unique_tensor(tau):
        stop("All entries of tau should be unique")

    if is_unsorted_tensor(tau):
        stop("Quantile values should be entered in ascending order")

    x_stds = x.std(dim=0, unbiased=True)
    if torch.any(x_stds == 0):
        stop("At least one of the x predictors has a std deviation of zero")

    g = torch.unique(groups).numel()

    if group_pen_factor is None:
        if norm == 2:
            counts = torch.bincount(groups)
            group_pen_factor = torch.sqrt(counts.float())
        else:
            group_pen_factor = torch.ones(g)

    if norm != 1 and norm != 2:
        stop("norm must be 1 or 2")

    if y.numel() != x.shape[0]:
        stop("length of x and number of rows in x are not the same")

    if weights is not None:
        if penalty == "ENet" | penalty == "Ridge":
            stop(
                "Cannot use weights with elastic net or ridge penalty. Can use it with lasso, though may be much slower than unweighted version."
            )
        if penalty == "aLASSO":
            # Warning
            print(
                "WARN: Weights are ignored when getting initial (Ridge) estimates for adaptive Lasso"
            )
        if weights.numel() != y.numel():
            stop("number of weights does not match number of responses")
        if sum(weights <= 0) > 0:
            stop("all weights most be positive")

    if y.ndim == 2:
        y = y.flatten()

    if group_pen_factor.min() < 0 or tau_penalty_factor.min() < 0:
        stop("Penalty factors must be non-negative.")

    if torch.sum(group_pen_factor) == 0 or torch.sum(tau_penalty_factor) == 0:
        stop(
            "Cannot have zero for all entries of penalty factors. This would be an unpenalized model"
        )

    dims = x.shape
    n = dims[0]
    p = dims[1]
    if groups.numel() != p:
        print(p, groups.numel())
        stop("length of groups is not equal to number of columns in x")

    if (weights is None) and penalty == "gAdLASSO":
        # Warning
        print("WARN: Initial estimate for group adaptive lasso ignores the weights")

    nt = tau.numel()
    na = a.numel()
    lpf = group_pen_factor.numel()
    if penalty != "gLASSO":
        a = getA(a, penalty)
    else:
        if a is not None:
            if a[0] != 1 | a.numel() != 1:
                stop(
                    "The tuning parameter a is not used for group lasso. Leave it set to NULL or set to 1"
                )
    if g == p:
        # Warning
        print("p groups for p predictors, not really using a group penalty")

    # I think this is repeated for no reason ? See ~line 66
    if y.ndim == 2:
        y = y.flatten()

    if not torch.is_tensor(x):
        stop("x must be a matrix")
    if penalty == "gLASSO" and norm == 1:
        stop(
            "Group Lasso with composite norm of 1 is the same as regular lasso, use norm = 2 if you want group lasso"
        )
    if norm == 1 and penalty == "gAdLASSO":
        # Warning
        print(
            "Group adapative lasso with 1 norm results in a lasso estimator where lambda weights are the same for each coefficient in a group. However, it does not force groupwise sparsity, there can be zero and non-zero coefficients within a group."
        )
    if norm == 2 and alg != "huber":
        stop("If setting norm = 2 then algorithm must be huber")
    if penalty == "gAdLASSO" and alg != "huber":
        # Warning
        print(
            "huber algorithm used to derive ridge regression initial estimates for adaptive lasso. Second stage of algorithm used lp"
        )
    if torch.any(tau <= 0) or torch.any(tau >= 1):
        stop("tau needs to be between 0 and 1")
    if torch.any(tau_penalty_factor <= 0) or torch.any(group_pen_factor < 0):
        stop("group penalty factors must be positive and tau penalty factors must be non-negative")
    if sum(group_pen_factor) == 0:
        stop("Some group penalty factors must be non-zero")
    if lpf != g:
        stop("group penalty factor must be of length g")

    if lamb is None:
        lamMax = R.getLamMaxGroup(
            x,
            y,
            groups,
            tau,
            group_pen_factor,
            gamma.item(),
            gamma_max=4,
            gamma_q=0.1,
            penalty="gLASSO",
            scalex=True,
            tau_penalty_factor=tau_penalty_factor,
            norm=norm,
            weights=weights,
        )[0]
        # lamMax = getLamMaxGroup(
        #     x,
        #     y,
        #     groups,
        #     tau,
        #     group_pen_factor,
        #     penalty=penalty,
        #     scalex=scalex,
        #     tau_penalty_factor=tau_penalty_factor,
        #     norm=norm,
        #     gamma=gamma,
        #     weights=weights,
        # )

        lamb = torch.exp(
            torch.linspace(
                np.log(lamMax),
                np.log(eps * lamMax),
                steps=nlambda,
            )
        )

    penalty_factor = torch.gather(group_pen_factor, 0, groups.long())

    if penalty == "gLASSO":
        return_val = R.rq_glasso(
            x.numpy(),
            y.numpy(),
            tau.numpy(),
            groups.numpy(),
            lamb.numpy(),
            group_pen_factor.numpy(),
            scalex,
            tau_penalty_factor.numpy(),
            max_iter,
            converge_eps,
            gamma.item(),
            lambda_discard,
            weights,
        )
    # if(penalty == "gLASSO"){
    # 	print("gAdLASSO case rq.group.pen")
    # 	return_val <- rq.glasso(x,y,tau,groups, lambda, group.pen.factor,scalex,tau.penalty.factor,max.iter,converge.eps,gamma,lambda.discard=lambda.discard,weights=weights,...)
    # }
    return return_val
