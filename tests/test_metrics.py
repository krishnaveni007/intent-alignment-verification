"""
Tests for src/evaluation/metrics.py
"""

import pytest

from src.evaluation.metrics import (
    task_success_rate,
    intent_error,
    misalignment_drift,
    regret,
    false_alignment_rate,
)


# ---------------------------------------------------------------------------
# task_success_rate
# ---------------------------------------------------------------------------


class TestTaskSuccessRate:
    def test_full_success(self):
        answers = {"car": "A", "hotel": "X"}
        correct = {"car": ["A", "B"], "hotel": ["X"]}
        assert task_success_rate(answers, correct) == 1.0

    def test_partial_success(self):
        answers = {"car": "A", "hotel": "Y"}
        correct = {"car": ["A"], "hotel": ["X"]}
        assert task_success_rate(answers, correct) == 0.5

    def test_no_success(self):
        answers = {"car": "Z"}
        correct = {"car": ["A"]}
        assert task_success_rate(answers, correct) == 0.0

    def test_empty_correct_ids(self):
        assert task_success_rate({"car": "A"}, {}) == 0.0

    def test_missing_answer_dimension(self):
        answers = {}
        correct = {"car": ["A"]}
        assert task_success_rate(answers, correct) == 0.0


# ---------------------------------------------------------------------------
# intent_error
# ---------------------------------------------------------------------------


class TestIntentError:
    def test_perfect_alignment(self):
        u = {"a": ["x", "y"]}
        assert intent_error(u, u) == 0.0

    def test_no_overlap(self):
        u_hat = {"a": ["x"]}
        u_star = {"a": ["y"]}
        assert intent_error(u_hat, u_star) == 1.0

    def test_partial_overlap(self):
        u_hat = {"a": ["x", "y"]}
        u_star = {"a": ["y", "z"]}
        # intersection={y}, union={x,y,z} -> jaccard dist = 1 - 1/3
        assert abs(intent_error(u_hat, u_star) - (1 - 1 / 3)) < 1e-9

    def test_both_empty(self):
        assert intent_error({}, {}) == 0.0

    def test_cosine_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            intent_error({"a": ["x"]}, {"a": ["x"]}, method="cosine")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            intent_error({"a": ["x"]}, {"a": ["x"]}, method="edit_distance")

    def test_multiple_dimensions_averaged(self):
        u_hat = {"a": ["x"], "b": ["p", "q"]}
        u_star = {"a": ["x"], "b": ["q", "r"]}
        # dim a: perfect -> 0.0
        # dim b: intersection={q}, union={p,q,r} -> 1 - 1/3
        expected = (0.0 + (1 - 1 / 3)) / 2
        assert abs(intent_error(u_hat, u_star) - expected) < 1e-9


# ---------------------------------------------------------------------------
# misalignment_drift
# ---------------------------------------------------------------------------


class TestMisalignmentDrift:
    def test_decreasing_drift(self):
        trajectory = [
            {},
            {"a": ["x"]},
        ]
        u_star = {"a": ["x"]}
        drift = misalignment_drift(trajectory, u_star)
        assert len(drift) == 2
        assert drift[0] > drift[1]
        assert drift[1] == 0.0

    def test_empty_trajectory(self):
        assert misalignment_drift([], {"a": ["x"]}) == []


# ---------------------------------------------------------------------------
# regret
# ---------------------------------------------------------------------------


class TestRegret:
    def test_optimal(self):
        assert regret("A", "A", ["A", "B"]) == 0.0

    def test_acceptable_not_best(self):
        assert regret("B", "A", ["A", "B"]) == 0.5

    def test_wrong_choice(self):
        assert regret("C", "A", ["A", "B"]) == 1.0

    def test_option_costs_optimal(self):
        costs = {"A": 10.0, "B": 12.0}
        # chosen == best -> cost diff = 0
        assert regret("A", "A", ["A", "B"], option_costs=costs) == 0.0

    def test_option_costs_higher(self):
        costs = {"A": 10.0, "B": 15.0}
        result = regret("B", "A", ["A", "B"], option_costs=costs)
        assert abs(result - 0.5) < 1e-9

    def test_option_costs_clamped(self):
        costs = {"A": 10.0, "B": 100.0}
        result = regret("B", "A", ["A", "B"], option_costs=costs)
        assert result <= 1.0

    def test_option_costs_missing_key_falls_back(self):
        # If a chosen_id is missing from costs, fall back to ordinal logic.
        costs = {"A": 10.0}
        result = regret("B", "A", ["A", "B"], option_costs=costs)
        assert result == 0.5


# ---------------------------------------------------------------------------
# false_alignment_rate
# ---------------------------------------------------------------------------


class TestFalseAlignmentRate:
    def test_basic(self):
        episodes = [
            {"task_success": True, "regret": 0.8},
            {"task_success": True, "regret": 0.2},
            {"task_success": False, "regret": 1.0},
        ]
        assert false_alignment_rate(episodes) == 0.5

    def test_no_successes(self):
        episodes = [{"task_success": False, "regret": 1.0}]
        assert false_alignment_rate(episodes) == 0.0

    def test_all_false_alignments(self):
        episodes = [
            {"task_success": True, "regret": 0.9},
            {"task_success": True, "regret": 0.7},
        ]
        assert false_alignment_rate(episodes) == 1.0

    def test_custom_threshold(self):
        episodes = [
            {"task_success": True, "regret": 0.4},
            {"task_success": True, "regret": 0.8},
        ]
        assert false_alignment_rate(episodes, regret_threshold=0.3) == 1.0
        assert false_alignment_rate(episodes, regret_threshold=0.9) == 0.0

    def test_empty_episodes(self):
        assert false_alignment_rate([]) == 0.0
