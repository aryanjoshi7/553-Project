import matplotlib.pyplot as plt
import numpy as np

# def scad_penalty(beta_hat, lambda_val, a_val):
#     is_linear = (np.abs(beta_hat) <= lambda_val)
#     is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
#     is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
#     linear_part = lambda_val * np.abs(beta_hat) * is_linear
#     quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
#     constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
#     return linear_part + quadratic_part + constant_part

def scad_penalty(beta, lambd, a):
    beta = np.abs(beta)
    penalty = np.zeros_like(beta)

    # region 1: |beta| <= lambda
    region1 = beta <= lambd
    penalty[region1] = lambd * beta[region1]

    # region 2: lambda < |beta| <= a * lambda
    region2 = (beta > lambd) & (beta <= a * lambd)
    penalty[region2] = (
        (-beta[region2]**2 + 2 * a * lambd * beta[region2] - lambd**2) / (2 * (a - 1))
    )

    # region 3: |beta| > a * lambda
    region3 = beta > a * lambd
    penalty[region3] = (lambd**2 * (a + 1)) / 2

    return penalty

plt_points = np.linspace(-20,20,100)
plt_y = []
for point in plt_points:
    plt_y.append(scad_penalty(point, 1.5, 3.7))
plt.plot(plt_points,np.array(plt_y))
plt.title("SCAD penalty, lambda = 1.5, a = 3.7")
plt.xlabel("l1 norm")
plt.ylabel("penalty")
plt.show()