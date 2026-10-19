import numpy as np

from greenqueue.experiments.e0 import shifting_saving


def test_shifting_saving_toy_case() -> None:
    ci = np.array([10.0, 10.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    s = shifting_saving(ci, d=1, w=2)
    assert abs(s[0] - 0.8) < 1e-12  # best start at hour 2: 2 vs 10
    assert (s >= 0).all()  # starting now is always allowed


def test_more_flexibility_never_saves_less() -> None:
    rng = np.random.default_rng(0)
    ci = rng.uniform(50, 300, 500)
    assert shifting_saving(ci, 4, 24).mean() >= shifting_saving(ci, 4, 6)[:470].mean() - 1e-12
