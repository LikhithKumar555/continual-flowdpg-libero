import pytest

from cfdpg.results import CellRecord
from cfdpg.runner import (
    already_done, build_eval_cmd, build_train_cmd, checkpoint_dir,
    episodes_flag, parse_eval_output, stage_episodes,
)

SAMPLE = (
    "INFO bot_eval.py:860 {'libero_object': {'avg_sum_reward': 0.82, 'avg_max_reward': 0.82, "
    "'pc_success': 82.0, 'n_episodes': 50, 'n_success': 41, 'pc_success_ci95': [69.2, 90.2]}}\n"
    "INFO bot_eval.py:860 {'avg_sum_reward': 0.82, 'avg_max_reward': 0.82, "
    "'pc_success': 82.0, 'n_episodes': 50, 'n_success': 41, 'pc_success_ci95': [69.2, 90.2], 'eval_s': 66.6}\n"
)


def test_parse_eval_output():
    assert parse_eval_output(SAMPLE) == (50, 41)


def test_parse_eval_output_missing():
    with pytest.raises(ValueError):
        parse_eval_output("nothing here")


def test_episodes_flag_sorted_unique():
    assert episodes_flag([5, 1, 5, 3]) == "[1,3,5]"
    with pytest.raises(ValueError):
        episodes_flag([])


def test_checkpoint_dir_padding():
    assert str(checkpoint_dir("/x/stage0", 10000)).endswith("checkpoints/010000/pretrained_model")
    assert str(checkpoint_dir("/x/stage0", 200)).endswith("checkpoints/000200/pretrained_model")


def test_stage_episodes_is_cumulative():
    te = {0: [3, 1], 1: [2], 2: [9, 8]}
    assert stage_episodes(te, [0, 1, 2], 0) == [1, 3]
    assert stage_episodes(te, [0, 1, 2], 1) == [1, 2, 3]
    assert stage_episodes(te, [2, 0, 1], 1) == [1, 3, 8, 9]  # permuted order


def test_eval_cmd_forces_resolution_and_checks_batch():
    cmd = build_eval_cmd(policy_path="p", suite="libero_object", task_id=2, out_dir="o", seed=0, n_episodes=50)
    assert "--env.observation_height=256" in cmd and "--env.observation_width=256" in cmd
    assert "--env.task_ids=[2]" in cmd
    with pytest.raises(ValueError):
        build_eval_cmd(policy_path="p", suite="s", task_id=0, out_dir="o", seed=0, n_episodes=45, batch_size=10)


def test_train_cmd_contains_core_flags():
    cmd = build_train_cmd(episodes=[2, 1], out_dir="o", seed=3, steps=100, batch_size=8,
                          num_layers=4, hidden_dim=512, num_heads=8, num_integration_steps=10)
    assert "--dataset.episodes=[1,2]" in cmd
    assert "--policy.objective=flow_matching" in cmd
    assert "--seed=3" in cmd and "--steps=100" in cmd and "--save_freq=100" in cmd


def test_already_done():
    r = CellRecord("run", "cum_bc", 0, "standard", 1, 0, "post", 50, 40)
    assert already_done([r], "run", 1, 0, "post")
    assert not already_done([r], "run", 1, 1, "post")
    assert not already_done([r], "other", 1, 0, "post")
