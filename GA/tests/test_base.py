from .setdata import X_diab, y_diab
import pytest
import GA

def test_output():
    result1 = GA.select(X_diab, y_diab)
    assert isinstance(result1, dict)
    assert 'selected' in result1.keys() and 'R2' in result1.keys() and 'R2pen' in result1.keys()
    
    result2 = GA.select(X_diab, y_diab, penalty=0.1)
    assert isinstance(result2, dict)

def test_bad_input():
    with pytest.raises(Error):
        GA.select(y_diab, X_diab)

        
