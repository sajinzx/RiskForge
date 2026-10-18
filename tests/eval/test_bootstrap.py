import numpy as np
import pandas as pd

from greenqueue.eval.bootstrap import block_bootstrap_ci, paired_saving_ci


def test_block_bootstrap_brackets_the_mean_and_widens_with_correlation() -> None:
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", periods=24 * 140, freq="h", tz="UTC")
    iid = rng.normal(10, 2, len(t))
    m, lo, hi = block_bootstrap_ci(iid, t, n_resamples=500)
    assert lo < m < hi and abs(m - 10) < 0.2
    # one level shift per week makes observations in a block strongly dependent -> wider interval
    weekly = np.repeat(rng.normal(0, 3, 20), 24 * 7) + rng.normal(0, 0.1, len(t))
    _, lo2, hi2 = block_bootstrap_ci(weekly, t, n_resamples=500)
    assert (hi2 - lo2) > (hi - lo)


def test_paired_saving_point_and_ci() -> None:
    base = np.full(100, 10.0)
    other = np.full(100, 7.0)
    blocks = np.repeat(np.arange(10), 10)
    s, lo, hi = paired_saving_ci(base, other, blocks, n_resamples=200)
    assert s == 0.3 and lo == hi == 0.3 or abs(lo - 0.3) < 1e-9 and abs(hi - 0.3) < 1e-9
