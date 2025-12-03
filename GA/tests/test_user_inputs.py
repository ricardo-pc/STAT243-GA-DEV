import numpy as np
import pandas as pd
import pytest
import GA

# Tests for X inputs
def test_X_input():
    with pytest.raises(TypeError):
        validate_inputs(X=5, y=np.array([1,2,3]))

    with pytest.raises(ValueError):
        validate_inputs(X=np.array([1,2,3]), y=np.array([1,2,3]))

# Tests for y inputs 
def test_y_input():
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=5)

    y_bad1 = pd.DataFrame({"a":[1,2], "b":[3,4]})
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=y_bad1)

    y_bad2 = np.array([[1,2],[3,4]])
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=y_bad2)

# Test for X and y must matching in length 
def test_same_length():
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4],[5,6]]), y=np.array([1,2]))

# Test for pred_names input
def test_pred_names_input():
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        pred_names="ht,dbh,sp")
    
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        pred_names=["ht","dbh","sp"])

# Tests for penalty & mut_rate inputs 
def test_penalty_and_mut_rate_inputs():
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        penalty="1%")
    
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        penalty=1.2)
    
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        mut_rate="1%")
    
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        mut_rate=1.2)

    with pytest.warns(RuntimeWarning):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        mut_rate=0.5)

# Tests for P and G inputs 
def test_population_and_generations():
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        P=12.5)
    
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        P=0.5)
        
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        G=12.5)
    
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        G=0.5)

# Tests for model_type input
def test_model_type_valid():
    with pytest.raises(ValueError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        model_type="lasso_regression")

# Tests for model_params input
def test_model_params_dict():
    with pytest.raises(TypeError):
        validate_inputs(X=np.array([[1,2],[3,4]]), y=np.array([1,2]),
                        model_params=[0.5,3])