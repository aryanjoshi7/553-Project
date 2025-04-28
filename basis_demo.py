import matplotlib.pyplot as plt
from patsy import dmatrix
import numpy as np

x = np.linspace(0, 10, 200)
X_spline = dmatrix("bs(x, df=6, degree=3, include_intercept=True)", {"x": x})
print(X_spline.shape)
plt.figure(figsize=(8, 4))
print(X_spline[:, 1])
for i in range(X_spline.shape[1]):
    plt.plot(x, X_spline[:, i], label=f"B{i+1}(x)")
plt.title("Cubic B-spline Basis Functions (df=6) (Degree=3)")
plt.legend()
plt.show() 

#x1 -> (x^3 + 2, x^3 + x^2 + x, )
#   model = f1(x1) + f1(x2) ....
# f1(x1)= theta_x1 * (b1(x1), b2(x1), b3(x1))
#
#
#