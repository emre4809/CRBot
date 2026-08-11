import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# --- Mock all sensor modules before importing env ---
mock_detect = MagicMock()
mock_detect.get_troops.return_value = []
mock_detect.model.names = {0: "giant", 1: "knight", 2: "archer"}
mock_detect.monitor = {"width": 555, "height": 780}

_default_state = {
    "elixir": 5,
    "hand": [
        {"name": "giant", "cost": 5},
        {"name": "knight", "cost": 3},
        {"name": "archer", "cost": 3},
        {"name": "arrows", "cost": 3},
    ],
    "playable_cards": ["giant", "knight", "archer", "arrows"],
}
mock_classify = MagicMock()
mock_classify.get_game_state.return_value = _default_state

mock_crowns = MagicMock()
mock_crowns.get_crown_counts.return_value = (0, 0)

mock_action = MagicMock()
mock_action.play_card.return_value = True
mock_action.ARENA_BOUNDS = (779, 473, 1251, 774)

sys.modules["detect"] = mock_detect
sys.modules["classify"] = mock_classify
sys.modules["crowns"] = mock_crowns
sys.modules["action"] = mock_action

from env import ClashRoyaleEnv, N_ACTIONS, N_GRID


# --- Tests ---

def make_env():
    return ClashRoyaleEnv()


def test_observation_shape():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (87,)


def test_action_space_size():
    env = make_env()
    assert env.action_space.n == 97


def test_elixir_normalized_in_obs():
    env = make_env()
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(0.5)  # elixir=5, /10=0.5


def test_do_nothing_skips_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    with patch("time.sleep"):
        env.step(96)  # do nothing
    mock_action.play_card.assert_not_called()


def test_affordable_card_calls_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    # card slot 1 (knight, cost 3), cell 0 → action = 1 * 24 + 0 = 24
    with patch("time.sleep"):
        env.step(24)
    mock_action.play_card.assert_called_once()


def test_unaffordable_card_skips_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    mock_classify.get_game_state.return_value = {
        "elixir": 2,
        "hand": [
            {"name": "giant", "cost": 5},
            {"name": "knight", "cost": 3},
            {"name": "archer", "cost": 3},
            {"name": "arrows", "cost": 3},
        ],
        "playable_cards": [],
    }
    with patch("time.sleep"):
        env.step(0)  # card 0 (giant), not in playable_cards
    mock_action.play_card.assert_not_called()
    mock_classify.get_game_state.return_value = _default_state


def test_episode_terminates_at_3_my_crowns():
    env = make_env()
    env.reset()
    mock_crowns.get_crown_counts.return_value = (3, 0)
    with patch("time.sleep"):
        _, _, terminated, _, _ = env.step(96)
    assert terminated is True
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_episode_terminates_at_3_enemy_crowns():
    env = make_env()
    env.reset()
    mock_crowns.get_crown_counts.return_value = (0, 3)
    with patch("time.sleep"):
        _, _, terminated, _, _ = env.step(96)
    assert terminated is True
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_step_returns_five_tuple():
    env = make_env()
    env.reset()
    with patch("time.sleep"):
        result = env.step(96)
    assert len(result) == 5  # obs, reward, terminated, truncated, info


def test_obs_dtype_is_float32():
    env = make_env()
    obs, _ = env.reset()
    assert obs.dtype == np.float32


def test_crowns_normalized_in_obs():
    env = make_env()
    mock_crowns.get_crown_counts.return_value = (2, 1)
    obs, _ = env.reset()
    assert obs[1] == pytest.approx(2 / 3)
    assert obs[2] == pytest.approx(1 / 3)
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_reset_survives_game_state_failure():
    env = make_env()
    mock_classify.get_game_state.side_effect = RuntimeError("boom")
    obs, info = env.reset()
    assert obs.shape == (87,)
    mock_classify.get_game_state.side_effect = None
    mock_classify.get_game_state.return_value = _default_state


def test_reset_survives_crown_counts_failure():
    env = make_env()
    mock_crowns.get_crown_counts.side_effect = RuntimeError("boom")
    obs, info = env.reset()
    assert obs.shape == (87,)
    mock_crowns.get_crown_counts.side_effect = None
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_step_survives_game_state_failure():
    env = make_env()
    env.reset()
    mock_classify.get_game_state.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        obs, reward, terminated, truncated, info = env.step(96)
    assert obs.shape == (87,)
    mock_classify.get_game_state.side_effect = None
    mock_classify.get_game_state.return_value = _default_state


def test_step_survives_get_troops_failure():
    env = make_env()
    env.reset()
    mock_detect.get_troops.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        obs, reward, terminated, truncated, info = env.step(96)
    assert obs.shape == (87,)
    mock_detect.get_troops.side_effect = None
    mock_detect.get_troops.return_value = []


def test_step_survives_crown_counts_failure():
    env = make_env()
    env.reset()
    mock_crowns.get_crown_counts.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        obs, reward, terminated, truncated, info = env.step(96)
    assert obs.shape == (87,)
    mock_crowns.get_crown_counts.side_effect = None
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_step_survives_play_card_failure():
    env = make_env()
    env.reset()
    mock_action.play_card.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        # action 24 = card slot 1 (knight, cost 3), affordable -> triggers play_card
        obs, reward, terminated, truncated, info = env.step(24)
    assert obs.shape == (87,)
    mock_action.play_card.side_effect = None
    mock_action.play_card.return_value = True
