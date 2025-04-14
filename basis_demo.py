import matplotlib.pyplot as plt
from patsy import dmatrix
import numpy as np

x = np.linspace(0, 10, 200)
X_spline = dmatrix("bs(x, df=6, degree=1, include_intercept=True) + x", {"x": x})
print(X_spline.shape)
plt.figure(figsize=(8, 4))
print(X_spline[:, 1])
for i in range(X_spline.shape[1]):
    plt.plot(x, X_spline[:, i], label=f"B{i+1}(x)")
plt.title("Cubic B-spline Basis Functions (df=6)")
plt.legend()
plt.show() 