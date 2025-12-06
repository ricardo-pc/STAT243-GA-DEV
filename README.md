# Genetic Algorithm (GA) for Variable Selection

**GA** is a Python package that implements a Genetic Algorithm (GA) to perform variable selection for regression tasks. It is designed to find the optimal predictors that maximizes model accuracy (measured by cross-validated $R^2$) while allowing for complexity penalties to favor sparse solutions.

## Installation

To install this package, clone the repository and install it via `pip` (editable mode):

```bash
git clone https://github.com/ricardo-pc/GA-dev.git
cd GA-dev
pip install -e .
```

## Quick Start / Demo

Here is a minimal example of how to use the `select` function with synthetic data.

```python
import numpy as np
from GA import select

# 1. Simulate data to test against a known truth 
np.random.seed(42)
n = 500 
p = 12

# Generate predictors (random normal distribution)
X = np.random.normal(size=(n, p))

# Assign non-zero coefficients to indices 0, 3, and 7
true_beta = np.zeros(p)
true_beta[0] = 2.0
true_beta[3] = -1.5
true_beta[7] = 1.5

# Add noise to the response
sigma = 0.5
eps = np.random.normal(scale=sigma, size=n)
y = X @ true_beta + eps

# Calculate Theoretical R^2 
signal_var = np.var(X @ true_beta)
theoretical_R2 = signal_var / (signal_var + sigma**2)

# 2. Run Variable Selection (Linear Regression)
# We expect the GA to find indices [0, 3, 7]
result = select(X, y, model_type="linear", G=50, P=20)

# 3. View and Compare Results
print(f"True Active Indices: [0, 3, 7]")
print(f"Selected Indices:    {sorted(result['selected'])}")
print("-" * 30)
print(f"Theoretical Max R2:  {theoretical_R2:.4f}")
print(f"GA Model CV R2:      {result['R2']:.4f}")
```

## Overview of Genetic Algorithm

Our Genetic Algorithm mimics natural selection to evolve a population of potential models toward the optimal solution.

1.  **Set Up:** Each candidate model (chromosome) is a binary vector where `1` indicates a predictor is included and `0` indicates it is excluded.
2.  **Initialization:** A population of size $P$ is randomly initialized.
3.  **Fitness Evaluation:** We evaluate the fitness of each chromosome using 5-fold Cross-Validated $R^2$.
    *   If a `penalty` ($\lambda$) is provided, the fitness is adjusted: $Fitness = R^2_{CV} - \lambda \times (\frac{k}{p})$, where $k$ is the number of selected variables and $p$ is the total number of variables.
4.  **Selection:** We use rank-based selection to determine which chromosomes become parents for the next generation.
5.  **Crossover:** We use single-point crossover to combine the features of two parents into new children.
6.  **Mutation:** Random mutations occur with a probability `mut_rate` (default 1%) to flip inclusion/exclusion status, adding diversity.

## API Documentation

The primary function of the package is `select`.

```python
select(X, y, pred_names=None, penalty=None, model_type="linear", 
       model_params=None, P=None, G=100, mut_rate=0.01)
```

Here is a clean, professional way to list the arguments while including the validation requirements (constraints) found in your code. Using a slight indentation or italics for the constraints keeps it readable without looking cluttered.

### Arguments

*   **`X`** : (2D Array or DataFrame)
    The predictor matrix (samples $\times$ features).
    *   *Must be 2-dimensional with at least 2 rows and 2 columns.*
    *   *Must match the number of rows in `y`.*

*   **`y`** : (1D Array, Series, or DataFrame)
    The response vector.
    *   *Must be 1-dimensional (if DataFrame, must have exactly 1 column).*

*   **`pred_names`** : (list, optional)
    Custom names for predictors (e.g., `["Age", "Height", ...]`).
    *   *Must be a list with length matching the number of columns in `X`.*
    *   *Defaults to generic names `x0`, `x1`, etc.*

*   **`penalty`** : (float, optional)
    Complexity penalty $\lambda$. Higher values favor fewer variables.
    *   *Must be between 0 and 1.*

*   **`model_type`** : (str)
    The underlying estimator to use.
    *   *Options: `"linear"` (default), `"lasso"`, or `"tree"`.*

*   **`model_params`** : (dict, optional)
    Hyperparameters for the underlying model (e.g., `{'max_depth': 3}`).
    *   *Must be a dictionary.*

*   **`P`** : (int, optional)
    Population size per generation.
    *   *Must be an integer > 1.*
    *   *Defaults to $\approx 1.5 \times$ number of predictors.*

*   **`G`** : (int)
    Number of generations to run.
    *   *Must be an integer > 1.*
    *   *Defaults to 100.*

*   **`mut_rate`** : (float)
    Mutation rate (probability of a bit flip).
    *   *Must be between 0 and 1.*
    *   *Defaults to 0.01 (1%). Warning issued if > 0.1.*

### Returns
A dictionary containing:
*   `"selected"`: Indices of the selected variables.
*   `"selected_names"`: Names of the selected variables.
*   `"R2"`: Unpenalized cross-validated $R^2$.
*   `"R2pen"`: Penalized cross-validated $R^2$.

---
## Test Scenarios

### Scenario 1: Standard Linear Regression (Baseline)

### Scenario 2: Sparse Linear Regression (With Penalty)

## Scenario 3: Lasso Regression (Double Selection)

## Scenario 4: Non-Linear Relationships (Decision Trees)


---

## Formal Testing

We use `pytest` for formal testing. To run the tests locally:

```bash
pytest tests/
```

## Team Members

*   **[Kea Madrone Hoiland Rutherford]** 
*   **[Lyla Jane Traylor]** 
*   **[Ricardo Pérez Castillo]** 

