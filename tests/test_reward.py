import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from reward import compute_reward


def test_neutral_step_no_reward():
    assert compute_reward(0, 0, 0, 0, [], False, False) == pytest.approx(0.0)


def test_my_crown_gain():
    assert compute_reward(0, 0, 1, 0, [], False, False) == pytest.approx(8.0)


def test_enemy_crown_gain():
    assert compute_reward(0, 0, 0, 1, [], False, False) == pytest.approx(-8.0)


def test_kill_reward():
    kills = [{"troop": "giant", "team": "enemy"}]
    assert compute_reward(0, 0, 0, 0, kills, False, False) == pytest.approx(0.1)


def test_win_terminal_bonus():
    # 3 crowns earned + win bonus
    result = compute_reward(0, 0, 3, 0, [], True, True)
    assert result == pytest.approx(8.0 * 3 + 15.0)


def test_loss_terminal_penalty():
    result = compute_reward(0, 0, 0, 3, [], True, False)
    assert result == pytest.approx(-8.0 * 3 - 15.0)


def test_no_terminal_reward_when_not_done():
    # even if won=True is passed, terminal reward only applies when terminated=True
    assert compute_reward(0, 0, 0, 0, [], False, True) == pytest.approx(0.0)
