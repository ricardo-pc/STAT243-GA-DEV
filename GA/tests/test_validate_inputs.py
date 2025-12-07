import numpy as np
import pandas as pd
import pytest
from GA.GA import _validate_inputs

# Tests for X inputs
def test_X_input():
    with pytest.raises(TypeError):
        _validate_inputs(X=5, y=np.array([1,2,3]),
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([1,2,3]), y=np.array([1,2,3]),
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2,3]]), y=np.array([1,2,3]),
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

# Tests for y inputs 
def test_y_input():
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=5,
                        pred_names=None, penalty=None, model_type="linear",
                        parent_selection="rank", crossover_type="single", 
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

    y_bad1 = pd.DataFrame({"a":[1,2], "b":[3,4]})
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=y_bad1,
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

    y_bad2 = np.array([[1,2],[3,4]])
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=y_bad2,
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

# Test for X and y must matching in length 
def test_same_length():
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4],[5,6]]), y=np.array([1,2]),
                        pred_names=None, penalty=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

# Test for pred_names input
def test_pred_names_input():
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                    pred_names="ht,dbh,sp", penalty=None, model_type="linear", 
                    parent_selection="rank", crossover_type="single",
                    model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                    pred_names=["ht","dbh","sp"], penalty=None, model_type="linear", 
                    parent_selection="rank", crossover_type="single",
                    model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

# Tests for penalty & mut_rate inputs 
def test_penalty_and_mut_rate_inputs():
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        penalty="1%", pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        penalty=1.2, pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)
    
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        mut_rate="1%", penalty=None, pred_names=None, 
                        parent_selection="rank", crossover_type="single",
                        model_type="linear", model_params=None, pop_gen=None, n_gen=100)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        mut_rate=1.2, penalty=None, pred_names=None, 
                        parent_selection="rank", crossover_type="single",
                        model_type="linear", model_params=None, pop_gen=None, n_gen=100)

    with pytest.warns(RuntimeWarning):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        mut_rate=0.5, penalty=None, pred_names=None, 
                        parent_selection="rank", crossover_type="single",
                        model_type="linear", model_params=None, pop_gen=None, n_gen=100)

# Tests for P and G inputs 
def test_population_and_generations():
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        pop_gen=12.5, penalty=None, pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, n_gen=100, mut_rate=0.01)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        pop_gen=-1, penalty=None, pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, n_gen=100, mut_rate=0.01)
        
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        n_gen=12.5, penalty=None, pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, mut_rate=0.01)
    
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]), 
                        n_gen=0, penalty=None, pred_names=None, model_type="linear", 
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, mut_rate=0.01)

# Tests for model_type input
def test_model_type_valid():
    with pytest.raises(ValueError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        model_type="lasso_regression", pred_names=None, penalty=None,
                        parent_selection="rank", crossover_type="single",
                        model_params=None, pop_gen=None, n_gen=100, mut_rate=0.01)

# Tests for model_params input
def test_model_params_dict():
    with pytest.raises(TypeError):
        _validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        model_params=[0.5,3], pred_names=None, penalty=None, 
                        parent_selection="rank", crossover_type="single",
                        model_type="linear", pop_gen=None, n_gen=100, mut_rate=0.01)