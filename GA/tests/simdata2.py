import numpy as np

# ====================================================
# simulate data with nonlinear relationships to be able to 
# test the results from the GA algorithm on a known truth 
# ====================================================
np.random.seed(42)
n = 500
p = 10

# Generate predictors
X_sim2 = np.random.normal(size=(n, p))

# Assign non-zero coefficients to only 3 predictors 
beta_nonlin = np.zeros(p)
beta_nonlin[1] = 2.0
beta_nonlin[4] = -3.5 
beta_nonlin[7] = 1.5 

# True predictors
true_preds_sim2 = [1, 4, 7]

# Add some noise 
sigma = 3.0

# Construct nonlinear signal part
signal = (
    beta_nonlin[1] * X_sim2[:, 1]**2 +
    beta_nonlin[4] * X_sim2[:, 4]**3 +
    beta_nonlin[7] * X_sim2[:, 7]**2
)

eps = np.random.normal(scale=sigma, size=n)
y_sim2 = signal + eps

# Theoretical R^2
signal_var = np.var(signal)
R2_sim2 = signal_var/(signal_var + sigma**2)