from chole_predict.analysis.metrics import rmse


def test_rmse_zero():
    assert rmse([1, 2], [1, 2]) == 0.0
