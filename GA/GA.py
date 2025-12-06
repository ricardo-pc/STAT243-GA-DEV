import numpy as np
import pandas as pd
import math
import warnings
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

def select(X, y, pred_names=None, penalty=None, model_type="linear", model_params=None, P=None, G=100, mut_rate=0.01):
    """Genetic Algorithm for variable selection.

    This function uses a genetic algorithm to do variable selection.
    The fitness metric is cross-validated R^2.

    Parameters
    ----------
    X: numpy array or pandas DataFrame 
        Predictor matrix (n samples by p predictors)
    y: numpy array or pandas DataFrame
        Response vector (length n samples)
    pred_names: list
        Optional input. Names of predictors. If None, generic names based on indexing are created.
    penalty: float
        Must be between 0 and 1. Complexity penalty. Default is None.
    model_type: str
        "linear" (for linear regression, default), "tree" (for decision tree), or "lasso" (for Lasso regression)
    model_params: dict
        Optional input. Model settings for decision tree or Lasso regression. Default is None.
    P: int 
        Must be > 1. Generation size. Default is ~1.5*p (where p is the number of predictors).
    G: int 
        Must be > 1. Number of generations. Default is 100. 
    mut_rate: float 
        Must be between 0 and 1. Mutation rate. Default is 0.01 (1%).
    parent_selection: str
        "rank" (for rank-based selection of parents), or "tournament" (for tournament style selciton of parents)
    crossover_type: str
        "single" (for single crossover point, default), or "double" (for double crossover point)

    Returns
    -------
    result: dict
        A dictonary with: selected (index of best predictors), selected_names (names of best predictors), R2 (R^2 value of best model), R2pen (penalized R^2 value)
    """
    # check that inputs are formatted as expected 
    _validate_inputs(X, y, pred_names, penalty, model_type, model_params, P, G, mut_rate)

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    n, p = X.shape

    # If user doesn't pass predictor names, create generic ones
    if pred_names is None:
        pred_names = [f"x{j}" for j in range(p)]
    else:
        pred_names = list(pred_names)

    # If user doesn't pass P, set P to be ~1.5*p 
    # (where p is chromosome length)
    if P is None:
        P = math.ceil(1.5*p)

    # Total sum of squares 
    SST = np.sum((y - np.mean(y))**2)

    # Main GA function
    best_chrom, best_main, best_aux = _run_ga(
        X, y, penalty, model_type, model_params,
        SST, P, G, mut_rate
    )

    if penalty is None:
        R2 = float(best_main)
        R2pen = float(best_main)
    else:
        R2pen = float(best_main)
        R2 = float(best_aux)

    selected = np.flatnonzero(best_chrom).tolist()
    selected_names = [pred_names[j] for j in selected]

    result = {
        "selected": selected,
        "selected_names": selected_names,
        "R2": R2,
        "R2pen": R2pen
    }

    return result


def _validate_inputs(X, y, pred_names, penalty, model_type, model_params, P, G, mut_rate):
    """
    Check for expected input types.
    """
    # X input
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be a numpy array or pandas DataFrame")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional matrix")

    if X.shape[0] <= 1 or X.shape[1] <= 1:
        raise ValueError("X must have more than 1 row and more than 1 column")

    # y input
    if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("y must be a numpy array or pandas Series/DataFrame")

    if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
        raise ValueError("y must have exactly one column")

    if isinstance(y, (np.ndarray, pd.Series)) and y.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    # X and y must have same length
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of rows")

    # pred_names input 
    if pred_names is not None:
        if not isinstance(pred_names, list):
            raise TypeError("pred_names must be a list")
    
        if len(pred_names) != X.shape[1]:
            raise ValueError(
                f"pred_names must have length {X.shape[1]} to match the number of predictors in X"
            )

    # penalty and mut_rate inputs 
    if penalty is not None:
        if not isinstance(penalty, float):
            raise TypeError("penalty must be a float")
        if not (0 <= penalty <= 1):
            raise ValueError("penalty must be between 0 and 1")

    if not isinstance(mut_rate, float):
        raise TypeError("mut_rate must be a float")
    if not (0 <= mut_rate <= 1):
        raise ValueError("mut_rate must be between 0 and 1")
    if mut_rate > 0.1:
        warnings.warn("mut_rate > 0.1 is very high", RuntimeWarning)

    # P and G inputs
    if P is not None:
        if not isinstance(P, int):
            raise TypeError("P must be an integer")
        if P <= 1:
            raise ValueError("P must be greater than 1")

    if not isinstance(G, int):
        raise TypeError("G must be an integer")
    if G <= 1:
        raise ValueError("G must be greater than 1")

    # model_type input
    valid_models = {"linear", "tree", "lasso"}
    if model_type not in valid_models:
        raise ValueError('model_type must be one of: "linear", "lasso", or "tree"')

    # model_params input
    if model_params is not None and not isinstance(model_params, dict):
        raise TypeError("model_params must be a dict")
    

def _run_ga(X, y, penalty, model_type, model_params, SST, P, G, mut_rate):
    """
    High-level Genetic Algorithm loop.
    """
    # K-fold splitter
    kf = KFold(n_splits=5, shuffle=True)
    folds = list(kf.split(X))

    # Create initial population (generation 0)
    # Each chromosome is a p-length vector of 0s and 1s
    # 1: include that predictor; 0: exclude that predictor 
    # Randomly choose 0 or 1 with probability 0.5 
    n, p = X.shape
    pop = (np.random.rand(P, p) < 0.5).astype(int)
    fitness_raw, fitness_pen = _compute_fitness(pop, X, y, penalty, model_type, model_params, SST, folds)

    if penalty is None:
        main_fit = fitness_raw
        aux_fit = fitness_pen
    else:
        main_fit = fitness_pen
        aux_fit = fitness_raw

    # Track the global best fitness and chromosome
    # over all generations
    best_idx = main_fit.argmax()
    best_main = main_fit[best_idx]
    best_aux = aux_fit[best_idx]
    best_chrom = pop[best_idx]

    # Loop over generations 
    for gen in range(G):
        
        # Evolve population (new generation)
        pop = _make_new_pop(pop, main_fit, mut_rate, parent_selection, crossover_type)

        # Evaluate fitness for new generation
        fitness_raw, fitness_pen = _compute_fitness(pop, X, y, penalty, model_type, model_params, SST, folds)

        if penalty is None:
            main_fit = fitness_raw
            aux_fit = fitness_pen
        else:
            main_fit = fitness_pen
            aux_fit = fitness_raw

        # Track best fitness for this generation
        best_idx_gen = main_fit.argmax()
        best_main_gen = main_fit[best_idx_gen]

        # If this generation found a better model, update global bests
        if best_main_gen > best_main:
            best_main = best_main_gen
            best_aux = aux_fit[best_idx_gen]
            best_chrom = pop[best_idx_gen]

    return best_chrom, best_main, best_aux

def _compute_fitness(gen, X, y, penalty, model_type, model_params, SST, folds):
    """
    Compute the fitness for each chromosome in the current population.
    Where fitness is cross-validated R^2. 
    """
    P, p = gen.shape
    fitness_raw = np.zeros(P)
    fitness_pen = np.zeros(P)

    # Default settings for each model
    default_tree = {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "random_state": 42
    }

    default_lasso = {
        "alpha": 0.001,
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": 42
    }

    # Loop over chromosomes 
    for i in range(P):
        chrom = gen[i]
        k = chrom.sum()

        # If no predictors selected, give bad fitness 
        if k == 0:
            fitness_raw[i] = -1e9
            fitness_pen[i] = -1e9
            continue

        X_sel = X[:, chrom == 1]
        SSR = 0.0 # sum of squared residuals

        # Cross-validation loop 
        for train_idx, test_idx in folds:

            X_train = X_sel[train_idx]
            y_train = y[train_idx]

            X_test = X_sel[test_idx]
            y_test = y[test_idx]

            if model_type == "linear":
                model = LinearRegression()

            elif model_type == "tree":
                params = default_tree if model_params is None else model_params
                model = DecisionTreeRegressor(**params)

            elif model_type == "lasso":
                params = default_lasso if model_params is None else model_params
                model = Lasso(**params)

            # Fit model on training fold 
            model.fit(X_train, y_train)

            # Predict on test fold
            errors = y_test - model.predict(X_test)
            SSR += np.sum(errors**2)

        R2_raw = 1 - (SSR/SST)  # Cross-validated R2

        if penalty is None:
            R2_pen = R2_raw
        else:
            R2_pen = R2_raw - penalty*(k/p)

        fitness_raw[i] = R2_raw
        fitness_pen[i] = R2_pen

    return fitness_raw, fitness_pen


def _make_new_pop(gen, fitness, mut_rate, parent_selection="rank", crossover_type = 'single'):
    """
    Create a new generation from the current one.
    """
    P, p = gen.shape
    pairs = P//2

    # 1. Parent selection to define which individuals breed
    if parent_selection == "rank":
        # Rank-based selection: rank determines breeding probability
        idx_sorted = np.argsort(fitness)
        ranks = np.empty(P, float)
        ranks[idx_sorted] = np.arange(1, P+1)
        selection_prob = ranks/ranks.sum()

        parent1_idx = np.random.choice(P, size=pairs, p=selection_prob)
        parent2_idx = np.random.randint(0, P, size=pairs)

    elif parent_selection == "tournament":
        # Tournament selection: partition population into groups each round
        # Determine tournament size (approximately sqrt(P))
        k = max(2, int(np.sqrt(P)))

        # Need P parents total to create new population
        num_parents_needed = P
    
        # Store selected parents
        selected_parents = []
        
        # Run tournament rounds until we have enough parents
        while len(selected_parents) < num_parents_needed:
            # Partition population into groups of size k
            num_groups = P // k
            remaining = P % k
            
            # Shuffle indices to randomly assign to groups
            shuffled_idx = np.random.permutation(P)
            
            # Run tournaments for each complete group (vectorized)
            complete_groups = shuffled_idx[:num_groups * k].reshape(num_groups, k)
            group_fitness = fitness[complete_groups]
            best_in_groups = complete_groups[np.arange(num_groups), np.argmax(group_fitness, axis=1)]
            selected_parents.extend(best_in_groups.tolist())
            
            # Handle remaining individuals if any (run tournament with smaller group)
            if remaining > 0 and len(selected_parents) < num_parents_needed:
                remaining_idx = shuffled_idx[num_groups * k:]
                remaining_fitness = fitness[remaining_idx]
                best_in_remaining = remaining_idx[np.argmax(remaining_fitness)]
                selected_parents.append(best_in_remaining)
        
        # Trim to exact number needed
        selected_parents = selected_parents[:num_parents_needed]
        
        # Randomly pair parents for breeding
        parent_pairs = np.random.permutation(selected_parents)
        parent1_idx = parent_pairs[0::2][:pairs]
        parent2_idx = parent_pairs[1::2][:pairs]
    
    parent1 = gen[parent1_idx]
    parent2 = gen[parent2_idx]

    # 2. Crossover to mix parents into children
    col_idx = np.arange(p)
    new_pop = np.zeros_like(gen)

    # crossover type
    if crossover_type == "single":
        # Single-point crossover
        cross_pts = np.random.randint(1, p, size=pairs)
        
        # create children 
        heads_mask = col_idx[np.newaxis, :] < cross_pts[:, np.newaxis]  # shape: (pairs, p)
        tails_mask = ~heads_mask

        # Create all children at once
        new_pop[0::2][:pairs] = parent1 * heads_mask + parent2 * tails_mask  # First children (even indices)
        new_pop[1::2][:pairs] = parent2 * heads_mask + parent1 * tails_mask  # Second children (odd indices)

    elif crossover_type == "double":
        # Double-point crossover
        cross_pt1 = np.random.randint(1, p-1, size=pairs)
        cross_pt2 = np.random.randint(cross_pt1 + 1, p, size=pairs)
        
        # create children
        # Middle segment is between the two crossover points
        heads_mask = col_idx[np.newaxis, :] < cross_pt1[:, np.newaxis]  # shape: (pairs, p)
        middle_mask = (col_idx[np.newaxis, :] >= cross_pt1[:, np.newaxis]) & (col_idx[np.newaxis, :] < cross_pt2[:, np.newaxis])
        tails_mask = col_idx[np.newaxis, :] >= cross_pt2[:, np.newaxis]

        # Create all children at once
        # First child: parent1 head + parent2 middle + parent1 tail
        new_pop[0::2][:pairs] = parent1 * heads_mask + parent2 * middle_mask + parent1 * tails_mask
        # Second child: parent2 head + parent1 middle + parent2 tail
        new_pop[1::2][:pairs] = parent2 * heads_mask + parent1 * middle_mask + parent2 * tails_mask

    # if P is odd, the last child is just a copy 
    # of the best-ranked (highest fitness) parent
    is_odd = (P % 2 == 1)
    if is_odd:
        best_parent_idx = fitness.argmax()
        new_pop[-1] = gen[best_parent_idx]

    # 3. Mutation: for each gene (bit), flip with small probability
    mutation_mask = (np.random.rand(P, p) < mut_rate)
    new_pop ^= mutation_mask

    return new_pop