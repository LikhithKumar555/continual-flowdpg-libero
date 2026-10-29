import numpy as np
import pytest

from cfdpg import metrics
from cfdpg.results import CellRecord, ResultStore, build_matrix

NAN = np.nan

# The toy example from the plan: 3 tasks.
TOY = np.array(
    [
        [0.9, NAN, NAN],
        [0.8, 0.9, NAN],
        [0.7, 0.8, 0.9],
    ]
)


def test_toy_fwt():
    assert metrics.forward_transfer_diag(TOY) == pytest.approx(0.90)


def test_toy_auc():
    # T1 = mean(0.9, 0.8, 0.7) = 0.80, T2 = 0.85, T3 = 0.90
    assert metrics.auc(TOY) == pytest.approx(0.85)


def test_toy_nbt():
    # T1: mean(0.1, 0.2) = 0.15, T2: 0.10 -> 0.125
    assert metrics.nbt(TOY, "diag") == pytest.approx(0.125)
    assert metrics.nbt(TOY, "peak") == pytest.approx(0.125)


def test_final_and_worst():
    assert metrics.final_average(TOY) == pytest.approx((0.7 + 0.8 + 0.9) / 3)
    assert metrics.worst_task(TOY) == pytest.approx(0.7)


def test_no_forgetting_gives_zero_nbt():
    r = np.array([[0.5, NAN], [0.5, 0.7]])
    assert metrics.nbt(r) == pytest.approx(0.0)


def test_backward_improvement_gives_negative_nbt():
    r = np.array([[0.5, NAN], [0.6, 0.7]])
    assert metrics.nbt(r) == pytest.approx(-0.1)


def test_peak_reference_differs_from_diag():
    # Task 0 improves after stage 1, then drops at stage 2.
    r = np.array(
        [
            [0.5, NAN, NAN],
            [0.8, 0.9, NAN],
            [0.6, 0.9, 0.9],
        ]
    )
    # diag: T0 drops = mean(0.5-0.8, 0.5-0.6) = -0.2 ; T1: 0.0 -> mean = -0.1
    assert metrics.nbt(r, "diag") == pytest.approx(-0.1)
    # peak: T0 stage1: ref=0.5 -> -0.3 ; stage2: ref=max(0.5,0.8)=0.8 -> 0.2 ; mean=-0.05
    #       T1: 0.0 -> overall = -0.025
    assert metrics.nbt(r, "peak") == pytest.approx(-0.025)


def test_nbt_needs_two_stages():
    with pytest.raises(ValueError):
        metrics.nbt(np.array([[0.9]]))


def test_missing_cell_is_rejected():
    bad = TOY.copy()
    bad[2, 1] = NAN
    with pytest.raises(ValueError, match="missing cell"):
        metrics.auc(bad)


def test_value_above_diagonal_is_rejected():
    bad = TOY.copy()
    bad[0, 2] = 0.1
    with pytest.raises(ValueError, match="above diagonal"):
        metrics.auc(bad)


def test_entry_success_accepts_vector_or_matrix():
    vec = np.array([0.0, 0.1, 0.3])
    assert metrics.entry_success(vec) == pytest.approx(0.4 / 3)
    assert metrics.entry_success(np.diag(vec)) == pytest.approx(0.4 / 3)


def test_summarize_keys():
    s = metrics.summarize(TOY, entry=np.array([0.0, 0.1, 0.2]))
    assert set(s) == {
        "fwt_diag", "auc", "final_avg", "worst_task", "nbt_diag", "nbt_peak", "entry_success"
    }


def test_aggregate_over_seeds():
    a = metrics.summarize(TOY)
    b = metrics.summarize(TOY)
    agg = metrics.aggregate_over_seeds([a, b])
    assert agg["auc"]["mean"] == pytest.approx(0.85)
    assert agg["auc"]["se"] == pytest.approx(0.0)
    assert len(agg["auc"]["per_seed"]) == 2


def test_build_matrix_roundtrip(tmp_path):
    store = ResultStore(tmp_path / "cells.jsonl")
    for stage in range(3):
        for task in range(stage + 1):
            rate = TOY[stage, task]
            store.append(
                CellRecord("run0", "cum_flowdpg", 0, "standard", stage, task, "post", 100, round(rate * 100))
            )
    mat = build_matrix(store.read(), 3)
    np.testing.assert_allclose(mat, TOY, equal_nan=True)


def test_duplicate_cell_is_rejected():
    rec = CellRecord("r", "m", 0, "standard", 0, 0, "post", 10, 5)
    with pytest.raises(ValueError, match="duplicate"):
        build_matrix([rec, rec], 2)


def test_entry_and_post_are_kept_apart():
    post = CellRecord("r", "m", 0, "standard", 1, 1, "post", 10, 9)
    entry = CellRecord("r", "m", 0, "standard", 1, 1, "entry", 10, 1)
    assert build_matrix([post, entry], 2, "post")[1, 1] == pytest.approx(0.9)
    assert build_matrix([post, entry], 2, "entry")[1, 1] == pytest.approx(0.1)


def test_invalid_records_are_rejected():
    with pytest.raises(ValueError):
        CellRecord("r", "m", 0, "standard", 0, 1, "post", 10, 5)      # task beyond stage
    with pytest.raises(ValueError):
        CellRecord("r", "m", 0, "standard", 1, 0, "entry", 10, 5)     # entry must be task == stage
    with pytest.raises(ValueError):
        CellRecord("r", "m", 0, "standard", 0, 0, "post", 10, 11)     # successes > rollouts
