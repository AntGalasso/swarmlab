"""The GNN arm (LLM side): G-Safeguard's defence as a `core.Defender`.

Each round it embeds what every agent wrote, runs the trained GNN on the conversation so far (the deployed pipeline of
upstream `defense_communication`), flags the agents whose attacker probability reaches the cut, and denies every
delivery of a flagged sender. A flag lasts until the conversation ends (upstream `set_role("attacker")` is never
undone); a new conversation starts when `obs.round_in_conversation` is 1 (`obs.t` when that is None).
"""
from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Sequence

import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn.private.checkpoint import load_checkpoint
from bazt_lab.features.gnn.private.embedding import shared_embedder
from bazt_lab.features.gnn.private.graph import family_key, predict_logits
from bazt_lab.features.gnn.private.train import checkpoint_path, training_finished

STAT = "gnn_logit"
FALLBACKS = ("pi_same_task", "none")


def deploy_family(attack: str, task: str, settings) -> str | None:
    """The GNN family run on the cell (attack, task): its own if trained (gnn_train.families), else per
    gnn.fallback_family: 'pi_same_task' -> insider_pi__<task>, 'none' -> None (refused). Unset -> core.Unset."""
    own = family_key(attack, task)
    if own in {family_key(*f) for f in core.get(settings, "gnn_train.families")}:
        return own
    fallback = core.get(settings, "gnn.fallback_family")
    if fallback not in FALLBACKS:
        raise ValueError(f"gnn.fallback_family must be one of {FALLBACKS}, got {fallback!r}")
    return family_key("insider_pi", task) if fallback == "pi_same_task" else None


class GNNArm:
    name = "GNN"
    recalibrated = False

    def __init__(self, embed: Optional[Callable[[Sequence[str]], torch.Tensor]] = None, model=None) -> None:
        self._embed, self._model = embed, model
        self.ctx = None

    @staticmethod
    def family_of(ctx) -> str:
        """The GNN family of the cell: its own (attack, task); infection and AiTM cells follow gnn.fallback_family."""
        cell = getattr(ctx, "cell", None)
        if not cell or "attack" not in cell or "task" not in cell:
            raise ValueError("the GNN arm needs ctx.cell = {'attack', 'task'} to pick the GNN of the cell")
        family = deploy_family(cell["attack"], cell["task"], ctx.settings)
        if family is None:
            raise ValueError(f"the GNN has no family for {family_key(cell['attack'], cell['task'])} and "
                             "gnn.fallback_family is 'none': this cell is refused")
        return family

    def reset(self, ctx: core.EpisodeContext) -> None:
        s = ctx.settings
        self.ctx = ctx
        self.cut = float(core.get(s, "gnn.cut"))
        self.carry = bool(core.get(s, "gnn.carry_flags"))
        # GNN_recalibrated: the cut is the calibrated logit threshold; absent (a calibration run) it is +inf
        self.logit_cut = float(ctx.thresholds.get(STAT, float("inf"))) if self.recalibrated else None
        if self._model is None:
            family = self.family_of(ctx)
            if not training_finished(s, family):
                raise RuntimeError(f"the GNN checkpoint of {family} at {checkpoint_path(s, family)} is missing or its "
                                   f"training did not finish; run train_unit for {family} again (it resumes)")
            self._model = load_checkpoint(checkpoint_path(s, family))[0]
        self._model.eval()
        if self._embed is None:
            self._embed = shared_embedder(s)
        self.history: List[List[List[str]]] = [[] for _ in range(ctx.B)]   # per episode: turns of N texts
        self.flagged = torch.zeros(ctx.B, ctx.N, dtype=torch.bool)

    def decide(self, obs: core.Observation) -> core.Actions:
        ctx = self.ctx
        texts = getattr(obs, "texts", None)
        if texts is None or len(texts) != ctx.B or any(len(row) != ctx.N for row in texts):
            raise ValueError(f"the GNN arm needs obs.texts: [B={ctx.B}][N={ctx.N}] str, the texts agents wrote this round")
        rnd = obs.round_in_conversation if obs.round_in_conversation is not None else obs.t   # t alone on the mock
        if rnd == 1:                                             # a new conversation: forget the last one
            self.history = [[] for _ in range(ctx.B)]
            self.flagged.zero_()
        logits = torch.zeros(ctx.B, ctx.N, dtype=torch.float64)
        for b in range(ctx.B):
            self.history[b].append(list(texts[b]))
            turns = self.history[b]
            emb = self._embed([t for turn in turns for t in turn]).view(len(turns), ctx.N, -1)
            logits[b] = predict_logits(self._model, ctx.adj[b], emb).to(torch.float64)
        hit = (logits > self.logit_cut) if self.recalibrated else (torch.sigmoid(logits) >= self.cut)
        hit &= obs.active
        self.flagged = (self.flagged | hit) if self.carry else hit
        deny = self.flagged[:, :, None] & obs.attempts
        self._fund(deny, logits, obs, ctx)
        return core.Actions(deny=deny, statistics={STAT: logits})

    @staticmethod
    def _fund(deny: torch.Tensor, logits: torch.Tensor, obs, ctx) -> None:
        """Drop denies the intervention budget cannot pay: most suspicious sender first, a sender that does not fit is
        skipped (in place)."""
        cost = float(ctx.costs["deny"])
        for b in range(deny.shape[0]):
            left = float(obs.remaining_intervention[b])
            if deny[b].sum().item() * cost <= left:
                continue
            for i in torch.argsort(logits[b], descending=True, stable=True).tolist():
                n = deny[b, i].sum().item()
                if n == 0:
                    continue
                if n * cost <= left:
                    left -= n * cost
                else:
                    deny[b, i] = False

    def state_dict(self) -> dict:
        return {"flagged": self.flagged.clone(), "history": [[list(t) for t in h] for h in self.history]}

    def load_state_dict(self, state: Mapping) -> None:
        self.flagged = state["flagged"].clone()
        self.history = [[list(t) for t in h] for h in state["history"]]


class GNNRecalibrated(GNNArm):
    """The same GNN with its cut recalibrated to alpha: acts when the logit exceeds ctx.thresholds['gnn_logit']."""
    name = "GNN_recalibrated"
    recalibrated = True
