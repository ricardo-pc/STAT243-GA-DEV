# ============================================================
# Genetic Algorithm for variable selection
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# 1. GA building blocks
# ------------------------------------------------------------

def initialize_population(P, p):
    """
    Create the initial population.

    Each individual is a length-p vector of 0/1:
      - 1 = include that predictor
      - 0 = exclude that predictor

    We just randomly choose 0 or 1 with probability 0.5.
    """
    population = (np.random.rand(P, p) < 0.5).astype(int)
    print("Initial population shape:", population.shape)
    print("Example chromosome (row 0):", population[0])
    return population


def compute_fitness_population(population, X, y, TSS, kf):
    """
    Compute the fitness for each chromosome in the population.

    Fitness = K-fold cross-validated R^2:
      1 - (sum of squared prediction errors across folds) / TSS

    If a chromosome selects no predictors, we punish it with a huge
    negative fitness so it basically never gets chosen.
    """
    P, p = population.shape
    fitness = np.zeros(P)

    # Loop over individuals
    for i in range(P):
        b = population[i, :]  # chromosome i

        # If no predictors selected, "bad" model
        if b.sum() == 0:
            fitness[i] = -1e9
            continue

        # Keep only the columns where b == 1
        X_sel = X[:, b == 1]

        # SSPE = sum of squared prediction errors over all CV folds
        SSPE = 0.0

        # K-fold CV loop
        for train_idx, test_idx in kf.split(X_sel):
            X_train = X_sel[train_idx, :]
            y_train = y[train_idx]
            X_test = X_sel[test_idx, :]
            y_test = y[test_idx]

            # Ordinary least squares linear regression
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions on test fold
            y_pred = model.predict(X_test)
            errors = y_test - y_pred
            SSPE += np.sum(errors ** 2)

        # R^2 with respect to the *original* TSS of y
        R2_cv = 1.0 - (SSPE / TSS)
        fitness[i] = R2_cv

    return fitness


def make_new_population(population, fitness, mutation_rate):
    """
    Create a new generation from the current one.

    Steps:
      1. Rank-based selection to define how likely each individual
         is to be chosen as a parent.
      2. Single-point crossover to mix two parents into children.
      3. Mutation: flip bits with small probability.

    This is the "evolution" step.
    """
    P, p = population.shape
    new_population = np.zeros_like(population)

    # 1. Rank-based selection:
    #    - Sort by fitness (low to high).
    #    - Higher fitness get higher rank.
    #    - Probabilities ∝ ranks (so best individuals are more likely).
    idx_sorted = np.argsort(fitness)  # indices from worst to best
    ranks = np.zeros(P, dtype=int)
    for r, idx_individual in enumerate(idx_sorted, start=1):
        ranks[idx_individual] = r
    total_rank = ranks.sum()
    selection_prob = ranks / total_rank

    child_count = 0
    while child_count < P:
        # Parent 1: chosen by rank-based probabilities
        parent1_index = np.random.choice(np.arange(P), p=selection_prob)
        parent1 = population[parent1_index]

        # Parent 2: chosen uniformly at random
        parent2_index = np.random.randint(P)
        parent2 = population[parent2_index]

        # 2. Single-point crossover:
        #    choose a point in [1, p-1] and swap tails of the chromosomes
        crossover_point = np.random.randint(1, p)

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]

        # 3. Mutation: for each gene (bit), flip with small probability.
        for j in range(p):
            if np.random.rand() < mutation_rate:
                child1[j] = 1 - child1[j]
            if np.random.rand() < mutation_rate:
                child2[j] = 1 - child2[j]

        # Add children to new population
        new_population[child_count, :] = child1
        child_count += 1

        if child_count < P:
            new_population[child_count, :] = child2
            child_count += 1

    return new_population


# ------------------------------------------------------------
# 2. Main GA driver (internal engine)
# ------------------------------------------------------------

def run_ga_variable_selection(X, y, predictor_names, TSS,
                              P=20, G=50, K=5,
                              mutation_rate=0.01, seed=42):
    """
    High-level GA loop.

    Inputs:
      - X, y, predictor_names, TSS: data pieces
      - P: population size
      - G: number of generations
      - K: folds for cross-validation
      - mutation_rate: probability of flipping each bit
      - seed: for reproducibility

    Output:
      - best_chromosome: 0/1 vector for the best model found
      - best_fitness: its CV-R^2
      - best_history: best fitness over generations (for plotting if you want)
    """
    np.random.seed(seed)

    # K-fold splitter (shuffled, fixed seed)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    n, p = X.shape  # not strictly needed, but nice to see

    # ---- Generation 0: random population + fitness ----
    population = initialize_population(P, p)
    fitness = compute_fitness_population(population, X, y, TSS, kf)

    print("\nGeneration 0 fitness values:")
    print(fitness)
    print("Best fitness in Gen 0:", fitness.max())
    print("Index of best individual:", fitness.argmax())
    print("Best chromosome in Gen 0:", population[fitness.argmax()])

    # Track the global best over *all* generations
    best_fitness_overall = fitness.max()
    best_chromosome_overall = population[fitness.argmax()].copy()
    best_history = [best_fitness_overall]

    print("\nInitial best fitness (Gen 0):", best_fitness_overall)

    # ---- Main GA loop ----
    for gen in range(G):
        print(f"\n=== Generation {gen + 1} ===")

        # Evolve population → new generation
        population = make_new_population(population, fitness, mutation_rate)

        # Evaluate new generation
        fitness = compute_fitness_population(population, X, y, TSS, kf)

        # Track best in this generation
        best_fitness_gen = fitness.max()
        best_index_gen = fitness.argmax()

        # If this generation found a better model, update global best
        if best_fitness_gen > best_fitness_overall:
            best_fitness_overall = best_fitness_gen
            best_chromosome_overall = population[best_index_gen].copy()

        best_history.append(best_fitness_overall)

        print("Best fitness this generation:", best_fitness_gen)
        print("Best overall fitness so far:", best_fitness_overall)

    return best_chromosome_overall, best_fitness_overall, best_history


# ------------------------------------------------------------
# 2.5 Public API: select()
# ------------------------------------------------------------

def select(X, y,
           predictor_names=None,
           P=20, G=50, K=5,
           mutation_rate=0.01, seed=42):
    """
    Main user-facing function.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Predictor matrix.
    y : array-like, shape (n_samples,)
        Response vector.
    predictor_names : array-like of strings, optional
        Names of predictors. If None, we create generic names.
    P, G, K, mutation_rate, seed : GA hyperparameters.

    Returns
    -------
    result : dict
        Dictionary with:
          - 'best_chromosome' : 0/1 array of selected predictors
          - 'best_R2'         : best CV-R^2
          - 'history'         : best R^2 over generations
          - 'selected_predictors' : list of predictor names
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n, p = X.shape

    # If user doesn't pass names, create generic ones
    if predictor_names is None:
        predictor_names = np.array([f"x{j}" for j in range(p)])
    else:
        predictor_names = np.asarray(predictor_names)

    # Compute TSS from y
    mean_y = np.mean(y)
    TSS = np.sum((y - mean_y) ** 2)

    # Run the internal GA engine
    best_chromosome, best_fitness, best_history = run_ga_variable_selection(
        X, y, predictor_names, TSS,
        P=P, G=G, K=K,
        mutation_rate=mutation_rate,
        seed=seed,
    )

    # Decode best chromosome into variable names
    selected_mask = best_chromosome == 1
    selected_predictors = predictor_names[selected_mask]

    result = {
        "best_chromosome": best_chromosome,
        "best_R2": best_fitness,
        "history": best_history,
        "selected_predictors": list(selected_predictors),
    }
    return result