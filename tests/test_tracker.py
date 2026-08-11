import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from tracker import TroopTracker, MIDLINE_Y


def test_enemy_classified_above_midline():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": MIDLINE_Y - 50, "conf": 0.9}]
    tracker.update(det, my_crowns=0, enemy_crowns=0)
    assert tracker.troops[0]["team"] == "enemy"


def test_ally_classified_below_midline():
    tracker = TroopTracker()
    det = [{"troop": "knight", "x": 100, "y": MIDLINE_Y + 50, "conf": 0.9}]
    tracker.update(det, my_crowns=0, enemy_crowns=0)
    assert tracker.troops[0]["team"] == "ally"


def test_no_new_troops_when_crown_fallen():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}]
    tracker.update(det, my_crowns=1, enemy_crowns=0)
    assert tracker.troops == []


def test_kill_on_third_missing_frame():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": MIDLINE_Y - 10, "conf": 0.9}]
    tracker.update(det, 0, 0)
    kills1 = tracker.update([], 0, 0)
    kills2 = tracker.update([], 0, 0)
    kills3 = tracker.update([], 0, 0)
    assert kills1 == [] and kills2 == []
    assert len(kills3) == 1 and kills3[0]["troop"] == "giant"
    assert tracker.troops == []


def test_position_updated_on_match():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    tracker.update([{"troop": "giant", "x": 115, "y": 108, "conf": 0.9}], 0, 0)
    assert tracker.troops[0]["x"] == 115
    assert tracker.troops[0]["y"] == 108


def test_far_detection_not_matched():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    # detection 200px away — should create a new troop, not match
    tracker.update([{"troop": "giant", "x": 300, "y": 100, "conf": 0.9}], 0, 0)
    assert len(tracker.troops) == 2


def test_ally_vanish_no_kill():
    tracker = TroopTracker()
    det = [{"troop": "knight", "x": 100, "y": MIDLINE_Y + 50, "conf": 0.9}]
    tracker.update(det, 0, 0)
    kills1 = tracker.update([], 0, 0)
    kills2 = tracker.update([], 0, 0)
    kills3 = tracker.update([], 0, 0)
    assert kills1 == [] and kills2 == [] and kills3 == []


def test_clear_resets_state():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    tracker.clear()
    assert tracker.troops == []
