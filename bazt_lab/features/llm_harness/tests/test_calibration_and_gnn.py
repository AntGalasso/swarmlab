"""Unit fields in rows, the GNN family per cell, D104 training questions, closed-loop calibration."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private.platform import NO_DEFENCE
from bazt_lab.features.llm_harness.private.seedrun import Env, SeedRun
from bazt_lab.features.llm_harness.tests.test_decisions import gnn_unit
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, N, Passive, arm, run, unit

TOLERANCE = 0.1                                  # calibration.tolerance of the test settings


@pytest.fixture()
def platform(families):
    return harness.LLMPlatform(families=lambda name: families[name])


# ------------------------------------------------------------------------------------------------ unit fields in rows
def test_every_unit_field_reaches_the_row(platform, tmp_path):
    rows, _ = run(platform, tmp_path, custom_axis=7, preset="good_classifier")
    assert all(r["preset"] == "good_classifier" and r["custom_axis"] == 7 for r in rows)
    plain, _ = run(platform, tmp_path / "p")
    assert all(r["condition"] is None and r["preset"] is None for r in plain)          # shared keys, None where absent


# ------------------------------------------------------------------------------------------------------- cell
def test_the_context_names_the_cell_of_the_batch_for_every_arm(platform, tmp_path):
    seen = []

    class Look(Passive):
        def reset(self, ctx):
            seen.append(ctx.cell)
            super().reset(ctx)

    for attack, task in (("infection", "mmlu"), ("none", "gsm8k"), ("insider_pi", "csqa")):
        platform.run_batch([unit(attack=attack, task=task, id=f"{attack}{task}")], arm(Look), SETTINGS,
                           core.CheckpointStore(tmp_path / attack), core.NullTracker())
    assert seen == [{"attack": "infection", "task": "mmlu"}, {"attack": "none", "task": "gsm8k"},
                    {"attack": "insider_pi", "task": "csqa"}]


# --------------------------------------------------------------------------------------- D104 training questions
def small_pool(task, seed, k, settings):
    """20 questions; their order is a function of the seed, so different seeds overlap."""
    order = core.uniforms(core.stream("pool", seed), (20,)).argsort().tolist()
    src = data.Source("x", None, None, None, None)
    return [data.TaskItem(task, f"{task}:{j}", f"Q{j}?", ("a", "b", "c", "d"), "A", src, {}) for j in order[:k]]


def test_gnn_training_never_draws_a_question_that_evaluation_or_calibration_uses():
    s = {**SETTINGS, "llm": {**SETTINGS["llm"], "conversations_per_cell": 6, "calibration_conversations_per_cell": 4}}
    banned = {i.id for seed in (1, 2) for i in small_pool("mmlu", seed, 3, s)}
    banned |= {i.id for seed in (101, 102) for i in small_pool("mmlu", seed, 2, s)}
    local = {**s, "runtime": {**s["runtime"], "n_insiders": 2, "random_edge_probability": 0.6}}
    for seed in range(8):
        u = {**gnn_unit(id=f"gx{seed}", conversations=4), "topology": "random_directed", "seed": seed,
             "audit": "uncapped", "intervention": "uncapped"}
        run_ = SeedRun(u, NO_DEFENCE, local, Env(None, None, small_pool, None), None, False, on_record=lambda r: None)
        got = [i.id for i in run_.items()]
        assert not set(got) & banned and len(got) == 4
        assert run_.evaluation_items() == banned


def test_evaluation_draws_are_unchanged_by_the_exclusion():
    u = {**unit(), "task": "mmlu", "seed": 3}
    run_ = SeedRun(u, arm(Passive), SETTINGS, Env(None, None, small_pool, None), None, False)
    assert [i.id for i in run_.items()] == [i.id for i in small_pool("mmlu", 3, 2, SETTINGS)]


# ----------------------------------------------------------------------------------------- closed-loop calibration
class Feedback(Passive):
    """Its statistic grows with the senders it has denied, so denying raises later statistics (like C does)."""
    name = "fb"

    def reset(self, ctx):
        super().reset(ctx)
        self.denied = 0.0

    def decide(self, obs):
        stat = core.uniforms(self.ctx.arm_stream(f"fb|{obs.t}", 0), (1, N, N)) + 0.1 * self.denied / N
        deny = obs.attempts & (stat > float(self.ctx.thresholds.get("fb", float("inf"))))
        self.denied += float(deny.any(2).sum())
        return core.Actions(deny=deny, statistics={"fb": stat})

    def state_dict(self):
        return {"denied": self.denied}

    def load_state_dict(self, state):
        self.denied = float(state["denied"])


def cal_units():
    return [unit(id=f"c{i}", attack="none", seed=100 + i, conversations=3) for i in range(4)]


def settings_mode(mode):
    return {**SETTINGS, "calibration": {"mode": mode, "tolerance": 0.1, "max_passes": 40}}


def realised(platform, tmp_path, thresholds, s):
    units = [{**u, "thresholds": thresholds} for u in cal_units()]
    rows = platform.run_batch(units, arm(Feedback), s, core.CheckpointStore(tmp_path), core.NullTracker())
    return sum(r["benign_restricted"] for r in rows) / sum(r["benign_deliveries"] for r in rows)


def test_the_calibration_mode_is_an_owner_key(platform, tmp_path):
    bare = {k: v for k, v in SETTINGS.items() if k != "calibration"}
    with pytest.raises(core.Unset, match=r"calibration\.mode"):
        platform.calibrate(cal_units(), arm(Feedback), bare, core.CheckpointStore(tmp_path))
    with pytest.raises(ValueError, match=r"calibration\.mode"):
        platform.calibrate(cal_units(), arm(Feedback), settings_mode("sideways"), core.CheckpointStore(tmp_path))


def test_open_loop_calibration_of_a_stateful_arm_denies_more_than_alpha_and_closed_loop_does_not(platform, tmp_path):
    alpha = SETTINGS["budgets"]["alpha"]
    open_s, closed_s = settings_mode("open_loop"), settings_mode("closed_loop")
    open_th = platform.calibrate(cal_units(), arm(Feedback), open_s, core.CheckpointStore(tmp_path / "o"))
    closed_th = platform.calibrate(cal_units(), arm(Feedback), closed_s, core.CheckpointStore(tmp_path / "c"))
    assert platform.last_calibration["mode"] == "closed_loop" and platform.last_calibration["within_tolerance"]
    assert realised(platform, tmp_path / "ro", open_th, open_s) > 1.25 * alpha                   # the defect
    rate = realised(platform, tmp_path / "rc", closed_th, closed_s)
    assert abs(rate - alpha) <= TOLERANCE * alpha, rate
    assert closed_th["fb"] > open_th["fb"] and platform.last_calibration["rate"] == pytest.approx(rate)


def test_closed_loop_is_deterministic_and_resumes_from_the_store(platform, tmp_path):
    s = settings_mode("closed_loop")
    store = core.CheckpointStore(tmp_path)
    first = platform.calibrate(cal_units(), arm(Feedback), s, store)
    passes = platform.last_calibration["passes"]
    again = platform.calibrate(cal_units(), arm(Feedback), s, store)
    other = platform.calibrate(cal_units(), arm(Feedback), s, core.CheckpointStore(tmp_path / "other"))
    assert first == again == other and platform.last_calibration["passes"] == passes >= 2


# ----------------------------------------------------- D104 with a pool smaller than the conversations (upstream repeats)
def tiny_pool(size):
    def sample(task, seed, k, settings):
        if k > size:
            raise ValueError(f"k={k} exceeds the {size} {task} records")
        order = core.uniforms(core.stream("tiny", seed), (size,)).argsort().tolist()
        src = data.Source("x", None, None, None, None)
        return [data.TaskItem(task, f"{task}:{j}", f"Q{j}?", ("a", "b", "c", "d"), "A", src, {}) for j in order[:k]]
    return sample


def training_run(sample, convs, unit_id="gt"):
    s = {**SETTINGS, "llm": {**SETTINGS["llm"], "conversations_per_cell": 4, "calibration_conversations_per_cell": 2}}
    local = {**s, "runtime": {**s["runtime"], "n_insiders": 2, "random_edge_probability": 0.6}}
    u = {**gnn_unit(id=unit_id, conversations=convs), "topology": "random_directed", "seed": 5, "audit": "uncapped",
         "intervention": "uncapped"}
    return SeedRun(u, NO_DEFENCE, local, Env(None, None, sample, None), None, False, on_record=lambda r: None)


def test_a_small_remaining_pool_is_drawn_from_with_repetition_keyed_and_never_from_the_excluded():
    run_ = training_run(tiny_pool(8), 12)
    got = [i.id for i in run_.items()]
    assert len(got) == 12 and not set(got) & run_.evaluation_items()
    assert len(set(got)) < 12 and set(got) <= {f"mmlu:{j}" for j in range(8)}
    assert got == [i.id for i in training_run(tiny_pool(8), 12).items()]                   # keyed
    assert run_.distinct_questions == len(set(got))


def test_an_empty_remaining_pool_refuses():
    with pytest.raises(ValueError, match="no question left"):
        training_run(tiny_pool(3), 5).items()


def test_the_training_log_records_the_distinct_question_count(platform, tmp_path, monkeypatch):
    from bazt_lab.features.gnn import public as gnn
    logged = []
    monkeypatch.setattr(gnn, "log_conversations", lambda settings, u, records: logged.extend(records))
    monkeypatch.setattr(gnn, "train_unit", lambda u, s: None)
    monkeypatch.setattr(gnn, "training_finished", lambda s, f: True, raising=False)
    local = {**SETTINGS, "runtime": {**SETTINGS["runtime"], "n_insiders": 2, "random_edge_probability": 0.6}}
    p = harness.LLMPlatform(families=platform._given[3], sample=tiny_pool(9))
    p.run_batch([gnn_unit(id="gl", conversations=6)], core.ArmSpec("GNN", Passive), local,
                core.CheckpointStore(tmp_path), core.NullTracker())
    assert len(logged) == 6 and all(1 <= r["distinct_questions"] <= 6 for r in logged)
    assert {r["distinct_questions"] for r in logged} == {len({r["question"] for r in logged})}


class Mixed(Feedback):
    """Always denies the edge 0 -> 1 (a fixed rule) besides the threshold denials, and says which ones the threshold caused."""
    name = "mixed"

    def decide(self, obs):
        stat = core.uniforms(self.ctx.arm_stream(f"fb|{obs.t}", 0), (1, N, N)) + 0.1 * self.denied / N
        by_threshold = obs.attempts & (stat > float(self.ctx.thresholds.get("fb", float("inf"))))
        rule = torch.zeros_like(by_threshold)
        rule[0, 0, 1] = True
        self.denied += float(by_threshold.any(2).sum())
        return core.Actions(deny=(by_threshold | rule) & obs.attempts, statistics={"fb": stat},
                            threshold_restricted=by_threshold)


def test_alpha_counts_only_the_restrictions_a_threshold_caused(platform, tmp_path):
    alpha = SETTINGS["budgets"]["alpha"]
    th = platform.calibrate(cal_units(), arm(Mixed), settings_mode("closed_loop"), core.CheckpointStore(tmp_path))
    assert abs(platform.last_calibration["rate"] - alpha) <= TOLERANCE * alpha
    rows = platform.run_batch([{**u, "thresholds": th} for u in cal_units()], arm(Mixed), settings_mode("closed_loop"),
                              core.CheckpointStore(tmp_path / "r"), core.NullTracker())
    all_restricted = sum(r["benign_restricted"] for r in rows) / sum(r["benign_deliveries"] for r in rows)
    assert all_restricted > alpha + 0.01                       # the fixed rule's denials are on top of alpha


def test_the_fallback_pass_never_denies_more_than_alpha_plus_tolerance(platform, tmp_path):
    class Jumpy(Feedback):
        name = "jumpy"

        def decide(self, obs):
            stat = core.uniforms(self.ctx.arm_stream(f"fb|{obs.t}", 0), (1, N, N)) + 3.0 * self.denied
            deny = obs.attempts & (stat > float(self.ctx.thresholds.get("fb", float("inf"))))
            self.denied += float(deny.any(2).sum())
            return core.Actions(deny=deny, statistics={"fb": stat})

    alpha = SETTINGS["budgets"]["alpha"]
    platform.calibrate(cal_units(), arm(Jumpy), settings_mode("closed_loop"), core.CheckpointStore(tmp_path))
    info = platform.last_calibration
    assert info["rate"] <= alpha * (1 + TOLERANCE) and info["passes"] <= 40


def test_events_without_a_batch_index_name_their_unit(platform, tmp_path):
    class Say(Passive):
        def decide(self, obs):
            ev = [{"event": "budget_exhausted", "budget": "audit"}] if obs.t == 1 else []
            return core.Actions(events=ev)

    rows, _ = run(platform, tmp_path, cls=Say)
    assert rows[0]["meta_events"] == [{"unit_id": "u1", "event": "budget_exhausted", "budget": "audit"}]
