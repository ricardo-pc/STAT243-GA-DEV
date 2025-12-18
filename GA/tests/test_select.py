from .setdata import X_diab, y_diab
import numpy as np
import pandas as pd
import pytest
import inspect
import GA

#########################################################
# Required tests for select function 
#########################################################
def test_output():
    result1 = GA.select(X_diab, y_diab)  # dataframe, series as inputs
    assert isinstance(result1, dict)
    assert 'selected' in result1.keys() and 'R2' in result1.keys() and 'R2pen' in result1.keys()

    # Check that can sum `selected` to get number of predictors selected.
    assert isinstance(np.sum(result1['selected']), (np.int64,np.int32,np.float64))
    
    result2 = GA.select(X_diab, y_diab, penalty=0.1)
    assert isinstance(result2, dict)
    assert 'selected' in result2.keys() and 'R2' in result2.keys() and 'R2pen' in result2.keys()

def test_req_args():
    sig = inspect.signature(GA.select)
    assert "penalty" in sig.parameters
    assert "pop_size" in sig.parameters
    assert "n_gen" in sig.parameters

def test_bad_input():
    with pytest.raises(TypeError):
        GA.select(y_diab, X_diab)


#########################################################
# Additional tests for select function outputs
#########################################################
X_diab_sample = X_diab.head(50)
y_diab_sample = y_diab.head(50)

def test_select_ouput():
    parent_methods = ["rank", "tournament"]
    crossover_types = ["single", "double"]
    model_types = ["linear", "lasso", "tree"]

    for parent_method in parent_methods:
        for crossover in crossover_types:
            for mtype in model_types:

                result = GA.select(
                    X=X_diab_sample,
                    y=y_diab_sample,
                    parent_selection=parent_method,
                    crossover_type=crossover,
                    model_type=mtype
                )

                assert isinstance(result, dict)
                assert len(result) == 3
                assert isinstance(result["selected"], np.ndarray)
                assert isinstance(result["R2"], float)
                assert isinstance(result["R2pen"], float)
                

#########################################################
# Additional tests for select function inputs
#########################################################
# Tests for X input
def test_X_input():
    with pytest.raises(TypeError):
        GA.select(X=pd.Series(np.random.rand(50)), y=np.random.rand(50))

    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(3,3), y=np.random.rand(3))
    
    X_missing=np.random.rand(20,5)
    X_missing[0, 0] = np.nan
    with pytest.raises(ValueError):
        GA.select(X=X_missing, y=np.random.rand(20))

# Tests for y input 
def test_y_input():
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(50,10), y=np.random.rand(50).tolist())

    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20,5))
    
    y_missing=np.random.rand(20)
    y_missing[0] = np.nan
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=y_missing)

    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(15))

# Test for parent_selection input
def test_parent_selection_input():
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            parent_selection="fit")

# Test for crossover_type input
def test_crossover_type_input():
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            crossover_type="one")

# Tests for penalty & mut_rate inputs 
def test_penalty_and_mut_rate_inputs():
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            penalty="1%")
    
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            penalty=1.2)
    
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            mut_rate="1%")
    
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            mut_rate=1.2)

    with pytest.warns(RuntimeWarning):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            mut_rate=0.5)

# Tests for pop_size and n_gen inputs 
def test_population_and_generations():
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            pop_size=12.5)
    
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            pop_size=-1)
    
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            n_gen=12.5)
    
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            n_gen=0)

# Tests for model_type input
def test_model_type_valid():
    with pytest.raises(ValueError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            model_type="lin")

# Tests for model_params input
def test_model_params_dict():
    with pytest.raises(TypeError):
        GA.select(X=np.random.rand(20,5), y=np.random.rand(20), 
            model_params=[0.5,3])