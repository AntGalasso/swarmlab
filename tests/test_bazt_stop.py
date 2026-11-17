from __future__ import annotations

import unittest
from unittest.mock import patch

import swarmlab as sl
from bazt.actions import ActionKind, Candidate
from bazt.config import BaztConfig
from bazt.ledger import BudgetLedger
from bazt.optimizer import BudgetedOptimizer, Situation
from bazt.scenarios import BudgetVisibleDecoySchedule


def _situation(inspect):
    return Situation(
        t=1,
        T=2,
        active=(0, 1),
        p_prime=[0.9, 0.9],
        deg=(1, 1),
        maxdeg=1,
        privileged=frozenset(),
        out_adj=((1,), (0,)),
        restricted=frozenset(),
        verdict_active=frozenset(),
        inspect=inspect,
    )


class StopFundingTests(unittest.TestCase):
    def test_stop_funding_check_reserves_shared_inspection_cost(self):
        cases = [
        (BaztConfig(budget_mode="pooled", B=1), False),
        (BaztConfig(budget_mode="pooled", B=2), True),
        (BaztConfig(budget_mode="tiered", K_hard=0, B_soft=1), False),
        (BaztConfig(budget_mode="tiered", K_hard=0, B_soft=2), True),
        (BaztConfig(budget_mode="dual_pct", audit_pct=0.1, intervention_pct=0.005), False),
        (BaztConfig(budget_mode="dual_pct", audit_pct=0.1, intervention_pct=0.02), True),
        ]
        for cfg, expected in cases:
            with self.subTest(mode=cfg.budget_mode, expected=expected):
                ledger = BudgetLedger(cfg, total_deliveries=100)
                optimizer = BudgetedOptimizer(cfg, sl.Config(N=2, T=2))
                self.assertIs(optimizer._can_fund_intervention_after_inspection(
                    _situation(lambda _node: None), ledger), expected)


    def test_stop_suppresses_eligible_inspection_when_intervention_is_unfundable(self):
        cfg = BaztConfig(budget_mode="dual_pct", audit_pct=0.1, intervention_pct=0.005, use_stop=True)
        sl_cfg = sl.Config(N=2, T=2)

        def high_value_inspection(node, p, H, C, config):  # noqa: ARG001
            return Candidate(ActionKind.INSPECT, node, -1, 10.0, 0.0, 10.0, 1, 10.0, p, H)

        inspections = []
        sit = _situation(lambda node: inspections.append(node))
        with patch("bazt.optimizer.score_inspect", high_value_inspection):
            plan = BudgetedOptimizer(cfg, sl_cfg).plan(sit, BudgetLedger(cfg, total_deliveries=100))

        self.assertTrue(plan.stop_triggered)
        self.assertTrue(plan.stop_suppressed_nodes)
        self.assertEqual(inspections, [])
        self.assertEqual(plan.inspected, {})

    def test_public_state_uses_binary_action_cost_and_tiered_pools(self):
        binary_tiered = BudgetLedger(BaztConfig(budget_mode="tiered", K_hard=1, B_soft=4,
                                                binary_actions=True), total_deliveries=100)
        binary_state = binary_tiered.public_state()
        self.assertEqual(binary_state["min_intervention_cost"], 3.0)
        self.assertTrue(binary_state["can_fund_any_intervention_after_inspection"])

        exhausted_tiered = BudgetLedger(BaztConfig(budget_mode="tiered", K_hard=0, B_soft=1),
                                        total_deliveries=100)
        exhausted_state = exhausted_tiered.public_state()
        self.assertFalse(exhausted_state["can_fund_any_intervention_after_inspection"])
        self.assertEqual(exhausted_state["intervention_remaining_by_pool"], {"hard": 0.0, "soft": 1.0})

    def test_pressure_trigger_is_a_finite_two_round_pulse(self):
        schedule = BudgetVisibleDecoySchedule("pressure", burst_duration=2, pressure_threshold=0.5)
        self.assertFalse(schedule(1, 0, {"pressure": 0.49}))
        self.assertTrue(schedule(2, 0, {"pressure": 0.5}))
        self.assertTrue(schedule(2, 1, {"pressure": 0.5}))
        self.assertTrue(schedule(3, 0, {"pressure": 0.7}))
        self.assertFalse(schedule(4, 0, {"pressure": 0.8}))
        self.assertEqual(schedule.trigger_round, 2)
        self.assertEqual(schedule.active_steps, [2, 3])

    def test_ledger_unfundable_trigger_is_separate_and_finite(self):
        schedule = BudgetVisibleDecoySchedule("stop_floor", burst_duration=2)
        self.assertFalse(schedule(1, 0, {"can_fund_any_intervention_after_inspection": True}))
        self.assertTrue(schedule(2, 0, {"can_fund_any_intervention_after_inspection": False}))
        self.assertTrue(schedule(3, 0, {"can_fund_any_intervention_after_inspection": False}))
        self.assertFalse(schedule(4, 0, {"can_fund_any_intervention_after_inspection": True}))
        self.assertEqual(schedule.trigger_round, 2)
        self.assertEqual(schedule.active_steps, [2, 3])

    def test_fixed_burst_ignores_budget_observations(self):
        schedule = BudgetVisibleDecoySchedule("fixed_time", burst_duration=2, fixed_round=3)
        self.assertFalse(schedule(2, 0, None))
        self.assertTrue(schedule(3, 0, None))
        self.assertTrue(schedule(4, 0, None))
        self.assertFalse(schedule(5, 0, None))
        self.assertEqual(schedule.active_steps, [3, 4])


if __name__ == "__main__":
    unittest.main()
