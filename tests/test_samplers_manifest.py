import numpy as np
import pytest

from cfdpg.manifest import DatasetManifest, TaskEntry
from cfdpg.samplers import CumulativeSampler

SIZES = [1000, 100, 10]   # deliberately unequal tasks


def _freq(sampler, n=60000):
    tasks, _ = sampler.sample(n)
    return np.bincount(tasks, minlength=sampler.n_tasks) / n


def test_uniform_follows_task_sizes():
    s = CumulativeSampler(SIZES, "uniform", seed=0)
    np.testing.assert_allclose(_freq(s), np.array(SIZES) / sum(SIZES), atol=0.01)


def test_balanced_is_equal_across_tasks():
    s = CumulativeSampler(SIZES, "balanced", seed=0)
    np.testing.assert_allclose(_freq(s), np.full(3, 1 / 3), atol=0.01)


def test_probabilities_match_mode():
    u = CumulativeSampler(SIZES, "uniform")
    b = CumulativeSampler(SIZES, "balanced")
    np.testing.assert_allclose(u.task_probabilities(), np.array(SIZES) / sum(SIZES))
    np.testing.assert_allclose(b.task_probabilities(), [1 / 3] * 3)


@pytest.mark.parametrize("mode", ["uniform", "balanced"])
def test_local_indices_stay_in_range(mode):
    s = CumulativeSampler(SIZES, mode, seed=1)
    tasks, local = s.sample(20000)
    assert (local >= 0).all()
    assert (local < np.array(SIZES)[tasks]).all()


@pytest.mark.parametrize("mode", ["uniform", "balanced"])
def test_every_transition_of_a_small_task_is_reachable(mode):
    s = CumulativeSampler([5, 5000], mode, seed=2)
    tasks, local = s.sample(200000)
    seen = set(local[tasks == 0].tolist())
    assert seen == set(range(5))


def test_add_task_grows_dataset():
    s = CumulativeSampler([100], "balanced", seed=0)
    s.add_task(50)
    assert s.n_tasks == 2
    np.testing.assert_allclose(_freq(s), [0.5, 0.5], atol=0.01)


def test_resume_gives_identical_stream():
    a = CumulativeSampler(SIZES, "uniform", seed=7)
    a.sample(64)
    state = a.get_state()
    expected = a.sample(64)

    b = CumulativeSampler(SIZES, "uniform", seed=999)
    b.set_state(state)
    got = b.sample(64)
    np.testing.assert_array_equal(expected[0], got[0])
    np.testing.assert_array_equal(expected[1], got[1])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        CumulativeSampler([], "uniform")
    with pytest.raises(ValueError):
        CumulativeSampler([10, 0], "uniform")
    with pytest.raises(ValueError):
        CumulativeSampler([10], "weighted")


def _entry(i, n=5000):
    return TaskEntry(task_id=i, name=f"task_{i}", n_episodes=50, n_transitions=n + i)


def test_manifest_is_cumulative_and_ordered():
    m = DatasetManifest("standard")
    for i in range(3):
        m.add_task(_entry(i))
    assert [t.task_id for t in m.at_stage(1)] == [0, 1]
    assert m.task_sizes(2) == [5000, 5001, 5002]
    with pytest.raises(ValueError):
        m.add_task(_entry(5))   # skipped ids


def test_manifest_save_load_and_tamper_detection(tmp_path):
    m = DatasetManifest("perm1")
    for i in range(2):
        m.add_task(_entry(i))
    path = tmp_path / "manifest.json"
    m.save(path)

    loaded = DatasetManifest.load(path)
    assert loaded.fingerprint() == m.fingerprint()
    assert loaded.order_name == "perm1"

    path.write_text(path.read_text().replace('"n_episodes": 50', '"n_episodes": 51', 1))
    with pytest.raises(ValueError, match="fingerprint"):
        DatasetManifest.load(path)
