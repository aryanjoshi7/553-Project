import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
n = 200
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

def f1(x): return np.sin(x)
def f2(x): return np.log1p(x)
y = f1(x1) + f2(x2) + np.random.normal(0, 0.3, size=n)

# Prepare design matrix
X = np.vstack((x1, x2)).T

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit a GAM: additive model with spline on each feature
gam = LinearGAM(s(0) + s(1)).fit(X_train, y_train)

# Plot partial dependence (the learned f1 and f2)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX), color='blue')
    ax.set_title(f"Estimated f{i+1}(x{i+1})")
    ax.set_xlabel(f"x{i+1}")
    ax.set_ylabel("Effect")
    ax.grid(True)

plt.tight_layout()
plt.show()
