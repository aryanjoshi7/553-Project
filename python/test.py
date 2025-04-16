import numpy as np
import torch
from utils import *
from scipy.stats import t
from workHorse import *
from rq_group_pen import *

R = R_ops(rqpen_path="~/code/EECS553/553-Project/rqpen", recompile=True)
n = 10
p = 3
x0 = np.random.randn(n, p)
X = np.hstack([x0, x0**2, x0**3])
X = X[:, np.argsort(np.tile(np.arange(1, p + 1), 3))]
y = (
    -2
    + X[:, 0]
    + 0.5 * X[:, 1]
    - X[:, 2]
    - 0.5 * X[:, 6]
    + X[:, 7]
    - 0.2 * X[:, 8]
    + t(df=2).rvs(n)
)
# The y vector (response variable)
X = torch.from_numpy(X)
y = torch.from_numpy(y)
# Create the group vector (repeat each integer 1:p, each 3 times)
group = torch.ones(X.shape[1])
tau = torch.tensor([0.5])
print(f"X shape {X.shape}, Y shape {y.shape}, Group Shape {group.shape}")


def test_getLamMaxGroup():
    R_lambda_max = R.getLamMaxGroup(
        X,
        y,
        group,
        tau,
        torch.ones(1),
        0.2,
        4,
        0.1,
        "gLASSO",
        True,
        torch.ones(tau.numel()),
        2,
        torch.ones(X.shape[0]),
    )
    py_lambda_max = getLamMaxGroup(
        X,
        y,
        group,
        tau,
        torch.ones(1),
        0.2,
        4,
        0.1,
        "gLASSO",
        True,
        torch.ones(tau.numel()),
        2,
        torch.ones(X.shape[0]),
    )

    print(f"lambda max from R: {R_lambda_max}, lambda max from Python: {py_lambda_max}")


def test_rq_group_pen():
    py_out = rq_group_pen(
        X,
        y,
        penalty="gSCAD",
        R=R
    )
    R_out = R.rq_group_pen(X, y, penalty="gSCAD")
    print(f"Py output: {py_out}")
    print(f"R output: {R_out}")
    

test_rq_group_pen()