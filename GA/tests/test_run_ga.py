import numpy as np
from GA.GA import _run_ga

def test_run_ga_output():
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    total_SS = np.sum((y - y.mean())**2)

    run_ga_result = _run_ga(X, y, parent_selection="rank",
                            crossover_type="single",
                            penalty=0.01, model_type="linear",
                            model_params=None, SST=total_SS,
                            pop_size=20, n_gen=10, mut_rate=0.01)

    assert isinstance(run_ga_result, tuple)
    assert len(run_ga_result) == 3
    assert isinstance(run_ga_result[1], np.floating)
    assert isinstance(run_ga_result[2], np.floating)
    assert isinstance(run_ga_result[0], np.ndarray)
    assert run_ga_result[0].shape == (5,)  # Match number of predictors
    assert np.all((run_ga_result[0] == 0) | (run_ga_result[0] == 1)) # Binary values only 
