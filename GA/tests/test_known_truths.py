from .simdata1 import X_sim1, y_sim1, true_preds_sim1, R2_sim1
from .simdata2 import X_sim2, y_sim2, true_preds_sim2, R2_sim2
import GA

def test_sim1_output():
    result_sim1 = GA.select(X_sim1, y_sim1, penalty=0.01)
    selected_sim1 = result_sim1["selected"]

    # Number of predictors between 2 and 6
    assert 2 <= len(selected_sim1) <= 6

    # At least 2 of the true predictors are selected 
    selected_set_sim1 = set(selected_sim1)
    num_correct_sim1 = len(selected_set_sim1.intersection(true_preds_sim1))
    assert num_correct_sim1 >= 2

    # R2 is reasonably close to theoretical R2
    assert abs(result_sim1["R2"] - R2_sim1) < 0.2

def test_sim2_output():

    sim_params = {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    result_sim2 = GA.select(X_sim2, y_sim2, model_type="tree",
        model_params=sim_params, penalty=0.01)
    selected_sim2 = result_sim2["selected"]

    # Number of predictors between 1 and 5
    assert 1 <= len(selected_sim2) <= 5

    # At least 2 of the true predictors are selected 
    selected_set_sim2 = set(selected_sim2)
    num_correct_sim2 = len(selected_set_sim2.intersection(true_preds_sim2))
    assert num_correct_sim2 >= 1

    # R2 is reasonably close to theoretical R2
    assert abs(result_sim2["R2"] - R2_sim2) < 0.2