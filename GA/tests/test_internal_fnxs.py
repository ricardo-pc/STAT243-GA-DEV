import numpy as np
from sklearn.model_selection import KFold
from GA.GA import _run_ga, _make_new_pop, _compute_fitness

#########################################################
# Test for _run_ga function 
#########################################################
def test_run_ga():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    ss_total = np.sum((y - y.mean())**2)

    run_ga_result = _run_ga(X, y, parent_selection="rank", crossover_type="single",
                            penalty=0.01, model_type="linear", model_params=None, 
                            SST=ss_total, pop_size=20, n_gen=10, mut_rate=0.01)

    assert isinstance(run_ga_result, tuple)
    assert len(run_ga_result) == 3
    assert isinstance(run_ga_result[1], np.floating)
    assert isinstance(run_ga_result[2], np.floating)
    assert isinstance(run_ga_result[0], np.ndarray)
    assert run_ga_result[0].shape == (5,)  # Match number of predictors
    assert np.all((run_ga_result[0] == 0) | (run_ga_result[0] == 1)) # Binary values only 


#########################################################
# Test for _make_new_pop function
#########################################################
def test_make_new_pop():
    # even-numbered pops 
    gen1 = np.random.randint(0,2, size =(10,5))
    fit1 = np.random.rand(10)

    parent_methods = ["rank", "tournament"]
    crossover_types = ["single", "double"]

    for parent_method in parent_methods:
        for crossover in crossover_types:

            new_pop_result1 = _make_new_pop(
                gen=gen1,
                fitness=fit1,
                mut_rate=0.01,
                parent_selection=parent_method,
                crossover_type=crossover
            )

            assert isinstance(new_pop_result1, np.ndarray)
            assert np.all((new_pop_result1 == 0) | (new_pop_result1 == 1)) # Binary only 
            assert new_pop_result1.shape == gen1.shape

    # odd-numbered pops 
    gen2 = np.random.randint(0,2, size =(11,5))
    fit2 = np.random.rand(11)

    new_pop_result2 = _make_new_pop(gen=gen2, fitness=fit2, mut_rate=0.01,
                                    parent_selection="rank", 
                                    crossover_type="single")

    assert new_pop_result2.shape == gen2.shape


#########################################################
# Tests for _compute_fitness function
#########################################################
# Test that outputs are correct type and shape
def test_compute_fitness_output():
    P, p = 10, 5
    gen = np.random.randint(0,2, size = (P,p))
    X = np.random.rand(50,p)
    y = np.random.rand(50)
    ss_total = np.sum((y - y.mean())**2)
    kf = KFold(n_splits=10, shuffle=True)
    folds = list(kf.split(X))

    model_types = ["linear", "lasso", "tree"]

    for model in model_types:

        fit_result1 = _compute_fitness(
            gen, X, y, 
            penalty=0.01,
            model_type=model, 
            model_params=None, 
            SST=ss_total, 
            folds=folds
        )

        assert isinstance(fit_result1, tuple)
        assert len(fit_result1) == 2
        assert isinstance(fit_result1[0], np.ndarray)
        assert isinstance(fit_result1[1], np.ndarray)
        assert fit_result1[0].shape == (P,)
        assert fit_result1[1].shape == (P,)

# Test the penalty functionality works
def test_compute_fitness_penalty():
    gen = np.random.randint(0,2, size = (10,5))
    gen[0] = np.ones(5, dtype = int) # at least one chrom has predictor selected
    X = np.random.rand(50,5)
    y = np.random.rand(50)
    ss_total = np.sum((y - y.mean())**2)
    kf = KFold(n_splits=10, shuffle=True)
    folds = list(kf.split(X))

    # Penalty reduces fitness 
    fitness_raw1, fitness_pen1 = _compute_fitness(gen, X, y, 
            penalty=0.1, model_type="linear", model_params=None, 
            SST=ss_total, folds=folds)

    has_predictors = gen.sum(axis=1) > 0
    assert np.all(fitness_pen1[has_predictors] <= fitness_raw1[has_predictors])

    # No penalty gives same fitness 
    fitness_raw2, fitness_pen2 = _compute_fitness(gen, X, y, 
            penalty=None, model_type="linear", model_params=None, 
            SST=ss_total, folds=folds)
    
    assert np.allclose(fitness_raw2, fitness_pen2)

# Test that chromosones with no predictors get bad fitness
def test_compute_fitness_bad_fitness():
    gen = np.zeros((5, 5), dtype=int)  # all chroms have no predictors
    X = np.random.rand(50,5)
    y = np.random.rand(50)
    ss_total = np.sum((y - y.mean())**2)
    kf = KFold(n_splits=10, shuffle=True)
    folds = list(kf.split(X))

    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, 
            penalty=None, model_type="linear", model_params=None, 
            SST=ss_total, folds=folds)
    
    assert np.all(fitness_raw < -1e8)
    assert np.all(fitness_pen < -1e8)