"""
Unit tests for the Fashion Stylist OpenEnv.
Run with: pytest tests/
"""
import pytest
from fashion_env.env import FashionStylistEnv
from fashion_env.models import Action
from fashion_env.tasks import grade_task_1, grade_task_2, grade_task_3, TASK_1_CUSTOMER, TASK_2_CUSTOMER, TASK_3_CUSTOMER
from fashion_env.inventory import INVENTORY_BY_ID


# ─────────────────────────── Fixtures ────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = FashionStylistEnv(task_id="task_casual_budget")
    env.reset()
    return env

@pytest.fixture
def medium_env():
    env = FashionStylistEnv(task_id="task_office_ready")
    env.reset()
    return env

@pytest.fixture
def hard_env():
    env = FashionStylistEnv(task_id="task_gala_stylist")
    env.reset()
    return env


# ─────────────────────────── Core API Tests ──────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert obs.step_number == 0
        assert obs.current_outfit == []
        assert obs.done is False
        assert len(obs.inventory) > 0

    def test_reset_clears_outfit(self, easy_env):
        easy_env.step(Action(action_type="add_item", item_id="T001"))
        obs = easy_env.reset()
        assert obs.current_outfit == []
        assert obs.step_number == 0


class TestStep:
    def test_add_item(self, easy_env):
        obs, reward, done, info = easy_env.step(
            Action(action_type="add_item", item_id="T001")
        )
        assert "T001" in obs.current_outfit
        assert "added" in info
        assert not done

    def test_add_nonexistent_item(self, easy_env):
        obs, reward, done, info = easy_env.step(
            Action(action_type="add_item", item_id="FAKE999")
        )
        assert "error" in info
        assert "FAKE999" not in obs.current_outfit

    def test_remove_item(self, easy_env):
        easy_env.step(Action(action_type="add_item", item_id="T001"))
        obs, reward, done, info = easy_env.step(
            Action(action_type="remove_item", item_id="T001")
        )
        assert "T001" not in obs.current_outfit

    def test_replace_item(self, easy_env):
        easy_env.step(Action(action_type="add_item", item_id="T001"))
        obs, reward, done, info = easy_env.step(
            Action(action_type="replace_item", item_id="T001", replacement_id="T002")
        )
        assert "T001" not in obs.current_outfit
        assert "T002" in obs.current_outfit

    def test_filter_inventory(self, easy_env):
        obs, reward, done, info = easy_env.step(
            Action(
                action_type="filter_inventory",
                filter_criteria={"category": "shoes", "max_price": 120.0}
            )
        )
        assert "filtered" in info
        for item in info["filtered"]:
            assert item["category"] == "shoes"
            assert item["price"] <= 120.0

    def test_finalize_outfit(self, easy_env):
        easy_env.step(Action(action_type="add_item", item_id="T001"))
        _, _, done, _ = easy_env.step(Action(action_type="finalize_outfit"))
        assert done is True

    def test_step_after_done_raises(self, easy_env):
        easy_env.step(Action(action_type="finalize_outfit"))
        with pytest.raises(RuntimeError):
            easy_env.step(Action(action_type="add_item", item_id="T001"))


class TestState:
    def test_state_returns_dict(self, easy_env):
        state = easy_env.state()
        assert isinstance(state, dict)
        assert state["task_id"] == "task_casual_budget"

    def test_state_tracks_outfit(self, easy_env):
        easy_env.step(Action(action_type="add_item", item_id="T001"))
        state = easy_env.state()
        assert "T001" in state["outfit_item_ids"]


class TestReward:
    def test_reward_range(self, easy_env):
        _, reward, _, _ = easy_env.step(
            Action(action_type="add_item", item_id="T001")
        )
        assert -1.0 <= reward.value <= 1.0

    def test_reward_has_breakdown(self, easy_env):
        _, reward, _, _ = easy_env.step(
            Action(action_type="add_item", item_id="T001")
        )
        assert "style_match" in reward.breakdown
        assert "color_harmony" in reward.breakdown
        assert "completeness" in reward.breakdown
        assert "budget" in reward.breakdown

    def test_reward_improves_with_complete_outfit(self, easy_env):
        obs, r1, _, _ = easy_env.step(Action(action_type="add_item", item_id="T004"))  # top
        obs, r2, _, _ = easy_env.step(Action(action_type="add_item", item_id="B002"))  # bottom
        obs, r3, _, _ = easy_env.step(Action(action_type="add_item", item_id="S001"))  # shoes
        # completeness should be maxed out now
        assert r3.breakdown["completeness"] >= r1.breakdown["completeness"]

    def test_over_budget_penalized(self, easy_env):
        # Add items that together exceed $200 budget
        easy_env.step(Action(action_type="add_item", item_id="O001"))  # $320
        easy_env.step(Action(action_type="add_item", item_id="T007"))  # $130
        easy_env.step(Action(action_type="add_item", item_id="S002"))  # $145
        _, reward, _, _ = easy_env.step(Action(action_type="add_item", item_id="A004"))  # $175
        assert reward.breakdown["budget"] < 0.5


# ─────────────────────────── Grader Tests ────────────────────────────────────

class TestGraders:
    def _items(self, ids):
        return [INVENTORY_BY_ID[i] for i in ids if i in INVENTORY_BY_ID]

    def test_task1_perfect_score(self):
        # Casual summer outfit within $200
        items = self._items(["T004", "B002", "S001"])  # ~$205 — slight over
        score = grade_task_1(items, TASK_1_CUSTOMER)
        assert 0.0 <= score <= 1.0

    def test_task1_empty_outfit_scores_zero(self):
        score = grade_task_1([], TASK_1_CUSTOMER)
        assert score == 0.0

    def test_task1_missing_shoes(self):
        items = self._items(["T004", "B002"])  # no shoes
        full_items = self._items(["T004", "B002", "S001"])
        score_partial = grade_task_1(items, TASK_1_CUSTOMER)
        score_full = grade_task_1(full_items, TASK_1_CUSTOMER)
        assert score_full > score_partial

    def test_task2_disliked_items_penalized(self):
        # Include a disliked item
        bad_items = self._items(["T008", "B001", "S002", "A006"])  # T008 is disliked
        good_items = self._items(["T001", "B001", "S002", "A006"])
        bad_score = grade_task_2(bad_items, TASK_2_CUSTOMER)
        good_score = grade_task_2(good_items, TASK_2_CUSTOMER)
        assert good_score > bad_score

    def test_task3_requires_4_items(self):
        three_items = self._items(["T005", "B007", "S007"])
        four_items  = self._items(["T005", "B007", "S007", "A003"])
        score_3 = grade_task_3(three_items, TASK_3_CUSTOMER)
        score_4 = grade_task_3(four_items, TASK_3_CUSTOMER)
        assert score_4 > score_3

    def test_all_graders_return_0_to_1(self):
        items = self._items(["T001", "B001", "S002"])
        for grader, customer in [
            (grade_task_1, TASK_1_CUSTOMER),
            (grade_task_2, TASK_2_CUSTOMER),
            (grade_task_3, TASK_3_CUSTOMER),
        ]:
            score = grader(items, customer)
            assert 0.0 <= score <= 1.0, f"{grader.__name__} out of range: {score}"


# ─────────────────────────── Step Limit Tests ────────────────────────────────

class TestStepLimit:
    def test_episode_ends_at_max_steps(self):
        env = FashionStylistEnv(task_id="task_casual_budget")
        env.reset()
        done = False
        for _ in range(20):  # max_steps is 15
            _, _, done, _ = env.step(Action(action_type="request_feedback"))
            if done:
                break
        assert done is True
