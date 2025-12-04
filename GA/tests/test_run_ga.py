import numpy as np
import pandas as pd
import pytest
from GA.GA import _run_ga

# Test output types and shapes are correct
def test_output_format():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    P = 10
    G = 5
    mut_rate = 0.01
    
    best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                              None, SST, P, G, mut_rate)
    
    assert best_chrom.shape == (5,)  # Should match number of predictors
    assert isinstance(best_main, (float, np.floating))
    assert isinstance(best_aux, (float, np.floating))

# Test best chromosone is binary
def test_best_chromosome_is_binary():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                              None, SST, 10, 5, 0.01)
    
    assert np.all((best_chrom == 0) | (best_chrom == 1))

# Test function runs with different population sizes
def test_different_population_sizes():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    for P in [10, 20, 50]:
        best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                                  None, SST, P, 5, 0.01)
        
        assert best_chrom.shape == (5,)

# Test function runs wth different number of generations
def test_different_generations():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    for G in [1, 5, 10]:
        best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                                  None, SST, 10, G, 0.01)
        
        assert best_chrom.shape == (5,)


# Test fitness values are finite
def test_fitness_values_finite():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                              None, SST, 10, 5, 0.01)
    
    assert np.isfinite(best_main)
    assert np.isfinite(best_aux)

# Test function works with penalty
def test_with_penalty():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    penalty = 0.1
    
    best_chrom, best_main, best_aux = _run_ga(X, y, penalty, "linear", 
                                              None, SST, 10, 5, 0.01)
    
    assert best_chrom.shape == (5,)
    assert np.isfinite(best_main)
    assert np.isfinite(best_aux)


# Test function works with different model types
def test_tree_model():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    best_chrom, best_main, best_aux = _run_ga(X, y, None, "tree", 
                                              None, SST, 10, 5, 0.01)
    
    assert best_chrom.shape == (5,)
    assert np.isfinite(best_main)

def test_lasso_model():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    
    best_chrom, best_main, best_aux = _run_ga(X, y, None, "lasso", 
                                              None, SST, 10, 5, 0.01)
    
    assert best_chrom.shape == (5,)
    assert np.isfinite(best_main)

# Test the best chormosome matches number of predictors in X
def test_chromosome_matches_predictors():
    for p in [3, 5, 10]:
        X = np.random.rand(50, p)
        y = np.random.rand(50)
        SST = np.sum((y - y.mean())**2)
        
        best_chrom, best_main, best_aux = _run_ga(X, y, None, "linear", 
                                                  None, SST, 10, 5, 0.01)
        
        assert best_chrom.shape == (p,)
