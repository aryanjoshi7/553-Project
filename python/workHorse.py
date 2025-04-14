import torch
from utils import stop, scale
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


def getA(a, penalty):
    if penalty in ["aLASSO", "gAdLASSO"]:
        if a is None:
            a = 1
        elif (torch.tensor(a) < 0).sum().item() > 0:
            raise ValueError('For adaptive lasso, the tuning parameter "a" must be positive')
    else:
        if a is None:
            if penalty in ["SCAD", "gSCAD"]:
                a = 3.7
                penalty = "SCAD"
            if penalty in ["MCP", "gMCP"]:
                a = 3
                penalty = "MCP"
        else:
            a_tensor = torch.tensor(a)
            if penalty == "SCAD" and (a_tensor <= 2).sum().item() > 0:
                raise ValueError('Tuning parameter "a" must be larger than 2 for SCAD')
            if penalty == "MCP" and (a_tensor <= 1).sum().item() > 0:
                raise ValueError('Tuning parameter "a" must be larger than 1 for MCP')
    return a


def rq_huber_deriv(r, tau, gamma):
    r = r.flatten()
    le_ind = torch.abs(r) <= gamma
    l_vec = torch.empty_like(r)
    if le_ind.numel() != 0:
        l_vec[le_ind] = (r[le_ind] / gamma + (2 * tau - 1)) / 2
        l_vec[~le_ind] = (torch.sign(r[~le_ind]) + (2 * tau - 1)) / 2
    else:
        l_vec = (torch.sign(r) + (2 * tau - 1)) / 2
    return l_vec


def neg_gradient(r, weights, tau, gamma, x, apprx):
    """
    Computes Huber-approximated negative gradient
    """
    huber_deriv = rq_huber_deriv(r, tau, gamma)
    wt_deriv = weights * huber_deriv
    if x.ndim == 1:
        return torch.mean(x * wt_deriv)
    else:
        return torch.mean(x * wt_deriv.view(-1, 1), dim=0)


def getLamMaxGroup(
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
    """
    Finds lambda max for group penalty.
    """
    if (
        (x is None)
        | (y is None)
        | (group_index is None)
        | (group_pen_factor is None)
        | (tau_penalty_factor is None)
    ):
        stop("getLamMaxGroup missing parameter")

    lambda_max = 0
    n = y.numel()
    if scalex:
        x = scale(x)
    if weights is None:
        weights = torch.ones(n)
    i = 0
    # make sure we can iterate over tau
    if isinstance(tau, float):
        tau = [tau]
    for tau_val in tau:
        pen_factor = group_pen_factor * tau_penalty_factor[i]
        validSpots = (pen_factor != 0)
        # TODO Not sure if npenVars is right
        mask__ = ~torch.isin(group_index, torch.nonzero(pen_factor != 0, as_tuple=False).squeeze())
        npenVars = torch.nonzero(mask__, as_tuple=False).squeeze()
        # npenVars = [j for j in range(len(group_index)) if group_index[j] not in validSpots]

        if len(npenVars) == 0:
            model = QuantReg(y.numpy(), np.ones((n, 1)))
            q1 = model.fit(q=tau_val, weights=weights.numpy())
            r = torch.tensor(y - q1.predict(), dtype=torch.float32)
        else:
            x_npen = x[:, npenVars]
            model = QuantReg(y.numpy(), np.c_[np.ones(n), x_npen.numpy()])
            q1 = model.fit(q=tau_val, weights=weights.numpy())
            r = torch.tensor(y - q1.predict(), dtype=torch.float32)

        gamma0 = min(gamma_max, max(gamma, torch.quantile(abs(r), gamma_q).float().item()))

        grad_k = -neg_gradient(r, weights, tau_val, gamma0, x, apprx="huber")
        grad_k_norm = {}
        for g in torch.unique(group_index):
            mask = (group_index == g)
            group_grad = grad_k[mask]
            grad_k_norm[int(g.item())] = torch.norm(group_grad, p=norm)
        
        norm_vals = torch.tensor([grad_k_norm[i.item()] for i in torch.unique(group_index)[validSpots].int()])
        # norm_vals = torch.tensor([grad_k_norm[int(group_index[i])] for i in validSpots])
        pen_vals = pen_factor[validSpots]
        lambda_candidate = torch.max(norm_vals / pen_vals).item()
        lambda_max = max(lambda_max, lambda_candidate)
        i += 1
    return lambda_max * 1.05  # no idea where the 1.05 comes from
