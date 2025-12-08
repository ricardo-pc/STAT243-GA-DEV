import numpy as np

# ====================================================
# simulate data to be able to test the results from 
# the GA algorithm on a known truth 
# ====================================================
np.random.seed(42)
n = 500 
p = 12

# Generate predictors
X_sim1 = np.random.normal(size=(n, p))

# Assign non-zero coefficients to only 3 predictors 
beta = np.zeros(p)
beta[0] = 2.0
beta[3] = -1.5
beta[7] = 1.5
beta[10] = 1.5

# True predictors
true_preds_sim1 = [0, 3, 7, 10]

# Add some noise
sigma = 0.25

# Generate response
eps = np.random.normal(scale=sigma, size=n)
y_sim1 = X_sim1 @ beta + eps

# Theoretical R^2
signal_var = np.var(X_sim1 @ beta)
R2_sim1 = signal_var/(signal_var + sigma**2)