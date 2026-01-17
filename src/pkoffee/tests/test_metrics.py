def test_check_size_match():
    import numpy as np
    from pkoffee.metrics import check_size_match, SizeMismatchError
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8])

    try:    
        check_size_match(a, b)
        assert True
    except SizeMismatchError:
        assert False

    try:    
        check_size_match(a, c)
        assert False
    except SizeMismatchError:
        assert True

def test_compute_r2():
    import numpy as np
    import pytest 
    from pkoffee.metrics import compute_r2

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_good = np.array([2.5, 0.0, 2.0, 8.0])
    y_pred_bad = np.array([0.0, 0.0, 0.0, 0.0])

    r2_good = compute_r2(y_true, y_pred_good)
    r2_bad = compute_r2(y_true, y_pred_bad)

    assert pytest.approx(r2_good, 0.01) == 0.9486
    assert r2_bad < 0

def test_compute_rmse():
    import numpy as np
    import pytest 
    from pkoffee.metrics import compute_rmse

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])

    rmse = compute_rmse(y_true, y_pred)

    assert pytest.approx(rmse, 0.0001) == 0.1000

def test_compute_mae():
    import numpy as np
    import pytest 
    from pkoffee.metrics import compute_mae

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])

    mae = compute_mae(y_true, y_pred)

    assert pytest.approx(mae, 0.0001) == 0.1000

