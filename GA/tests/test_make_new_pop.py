import numpy as np
import pandas as pd
import pytest
from GA.GA import _make_new_pop

# Test output shape matches input shape
def test_output_shape():
    gen = np.random.randint(0,2, size =(10,5))
    fitness = np.random.rand(10)
    mut_rate = 0.01

    new_pop = _make_new_pop(gen, fitness, mut_rate)

    assert new_pop.shape == gen.shape

def test_output_shape_odd_pop():
    gen = np.random.randint(0,2, size =(11,5))
    fitness = np.random.rand(11)
    mut_rate = 0.01

    new_pop = _make_new_pop(gen, fitness, mut_rate)

    assert new_pop.shape == gen.shape

# Test output only include binary values
def test_output_binary():
    gen = np.random.randint(0,2, size =(10,5))
    fitness = np.random.rand(10)
    mut_rate = 0.01

    new_pop = _make_new_pop(gen, fitness, mut_rate)
    
    assert np.all((new_pop == 0) | (new_pop == 1))

# Test input type matches output type
def test_output_type():
    gen = np.random.randint(0,2, size =(10,5))
    fitness = np.random.rand(10)
    mut_rate = 0.01

    new_pop = _make_new_pop(gen, fitness, mut_rate)

    assert isinstance(new_pop, np.ndarray)

# Test different population sizes
def test_diff_pop_size():
    for P in [4, 10, 20, 50]:
        gen = np.random.randint(0,2, size =(P,8))
        fitness = np.random.rand(P)
        mut_rate = 0.01

        new_pop = _make_new_pop(gen, fitness, mut_rate)

        assert new_pop.shape == (P,8)

# Test different number of predictors
def test_diff_pop_size():
    for p in [3, 5, 10, 20]:
        gen = np.random.randint(0,2, size =(10,p))
        fitness = np.random.rand(10)
        mut_rate = 0.01

        new_pop = _make_new_pop(gen, fitness, mut_rate)

        assert new_pop.shape == (10,p)

