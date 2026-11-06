"""What each action is worth: expected loss reduction minus weighted utility loss (Poolsappasit, Dewri & Ray, TDSC 2012,
section 5; the damage, response and operational cost classes of Lee et al. 2000), priced on the contagion model.

Notation. B episodes, N agents. At round t of T, `hops` = T - t further rounds can still spread what is delivered now
(the delivery of this round is the first hop), and `rounds_left` = T - t + 1 rounds still have deliveries to pay for.
r[b, i] is the posterior probability that agent i is compromised. What is at stake if i is compromised is its reach:
  node[i]      d * H_i                  sum of the susceptibility weights it can still reach (truncated Katz)
  edge[i, j]   d * R_j                  the part behind the single edge i -> j (H_i = sum_j A_ij R_j)
  tail[i, j]   d * (R_j - R_j^(K-1))    what a one-round delay of that delivery removes
with the susceptibility weight s_j = (1 - r_j) w_j (w = sink_weight at the sink, 1 elsewhere) and d the Katz decay of the
episode, bazt.katz_decay / rho(adj) (D115, `katz_decay`): one value per episode from its base graph, the same at every
round and for every counterfactual graph (the reinstated one included), so d x rho <= bazt.katz_decay < 1 on all of them.
The structural share of a node or an edge is its relative eigen-drop (NetShield, NetMelt) times the spread that does not
start at the node itself, so a benign hub on the paths of other sources has value too.

Costs are in platform units: `costs[kind]` per affected delivery, agent-level actions per delivery they remove
(out-degree x rounds_left, the rule of core.Actions). The out-degree of a cost counts every live delivery of the agent
(base edge, both ends active), an edge quarantined earlier included: the mock platform charges isolation that way, and
for degradation and rate limit it is the dearer reading. A persistent edge quarantine is priced per delivery it removes
over the remaining rounds, also the dearer reading, so a plan never undershoots what the platform charges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Tuple

import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import contagion as ct
from bazt_lab.features.bazt_slot.private.params import Params

Tensor = torch.Tensor
K = core.ActionKind


@dataclass(frozen=True)
class World:
    """What the model sees at round t. All tensors have B as leading dimension."""
    t: int
    T: int
    adj: Tensor           # bool [B, N, N]  base graph (A's allowlist)
    live: Tensor          # bool [B, N, N]  deliveries that can still happen: both ends active, edge not quarantined
    quarantined: Tensor   # bool [B, N, N]
    active: Tensor        # bool [B, N]
    isolated: Tensor      # bool [B, N]
    degraded: Tensor      # bool [B, N]     every outgoing delivery is sanitized
    limited: Tensor       # bool [B, N]     reaches at most `rate_limit` receivers
    r: Tensor             # float64 [B, N]  posterior P(compromised)
    sink: Tensor          # int64 [B]
    decay: Tensor | None = None   # [B] Katz decay of each episode (`katz_decay`); None: computed from `adj`


@dataclass(frozen=True)
class Reach:
    node: Tensor          # [B, N]
    edge: Tensor          # [B, N, N]
    tail: Tensor          # [B, N, N]
    spread: Tensor        # [B]  sum_i r_i node_i over active agents
    node_rel: Tensor      # [B, N]
    edge_rel: Tensor      # [B, N, N]
    pair: Tuple[Tensor, Tensor] | None = None     # leading right / left eigenvectors, to warm-start the next round


@dataclass(frozen=True)
class Priced:
    net: Tensor           # expected loss reduction minus weighted utility loss (raw, valid or not)
    cost: Tensor          # platform units
    recurrence: Tensor    # rounds the action is paid again if it is repeated (the multiplier of its benign price)
    valid: Tensor         # bool: eligible, affordable in kind, and net > 0


def hops(w: World) -> int:
    return max(w.T - w.t, 0)


def rounds_left(w: World) -> int:
    return w.T - w.t + 1


def katz_decay(adj: Tensor, p: Params) -> Tensor:
    """float64 [B]: the Katz decay of each episode, bazt.katz_decay / spectral radius of its base graph (D115). BAZT
    computes it once per episode at reset and hands it to every World."""
    return ct.relative_decay(adj, p.katz_fraction, p.iterations)


def decay(w: World, p: Params) -> Tensor:
    """[B] in the dtype of the posterior: the decay the World carries, or the one of its base graph."""
    d = katz_decay(w.adj, p) if w.decay is None else w.decay
    return d.to(dtype=w.r.dtype, device=w.r.device)


def unit_cost(costs: Mapping[str, float], kind: core.ActionKind) -> float:
    try:
        return float(costs[kind.value])
    except KeyError:
        raise core.Unset(f"costs.<main|robustness>.{kind.value}") from None


def out_degree(mask: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
    return mask.sum(-1).to(dtype)


def centrality(w: World) -> Tensor:
    """float64 [B, N]: how much the task needs agent i: its share of the busiest sender's degree and the share of the
    horizon still ahead (the form of swarmlab.centrality, on the out-degree of the base graph)."""
    deg = out_degree(w.adj, w.r.dtype)
    return 0.5 * deg / deg.amax(-1, keepdim=True).clamp_min(1.0) + 0.5 * (1.0 - w.t / w.T)


def transmission(w: World, p: Params) -> Tensor:
    """float64 [B, N, N]: live edges scaled by how much of the sender's traffic still carries a payload."""
    live = w.live.to(w.r.dtype)
    out = out_degree(w.live, w.r.dtype)
    rate = torch.where(w.limited & (out > p.rate_limit), p.rate_limit / out.clamp_min(1.0), torch.ones_like(out))
    factor = (1.0 - p.sanitize_success * w.degraded.to(w.r.dtype)) * rate
    return live * factor[..., None]


def _susceptibility(w: World, p: Params, everyone: bool = False) -> Tensor:
    weight = torch.ones_like(w.r)
    weight[torch.arange(w.r.shape[0], device=w.r.device), w.sink] = p.sink_weight
    s = (1.0 - w.r) * weight
    return s if everyone else s * w.active.to(w.r.dtype)


def reach(w: World, A: Tensor, p: Params, topology: bool, pair: Tuple[Tensor, Tensor] | None = None,
          structure: bool = True) -> Reach:
    """The contagion model on the transmission matrix `A`. Without topology each agent stands for one unit at stake and
    the graph does not matter, but nothing is at stake once no round is left. `structure=False` skips the eigenpair
    (the audit only needs the reach)."""
    K_ = hops(w)
    zero_edge = torch.zeros_like(A)
    if not topology:
        on = float(K_ > 0)
        node = torch.full_like(w.r, on)
        return Reach(node, A * on, A * float(K_ == 1), (w.r * node * w.active).sum(-1), torch.zeros_like(w.r), zero_edge)
    s = _susceptibility(w, p)
    d = decay(w, p)
    H, R, _ = ct.katz(A, s, d, K_)
    _, R_short, _ = ct.katz(A, s, d, K_ - 1)
    node = d[:, None] * H
    spread = (w.r * node * w.active).sum(-1)
    edge, tail = d[:, None, None] * A * R[:, None, :], d[:, None, None] * A * (R - R_short)[:, None, :]
    if not structure:
        return Reach(node, edge, tail, spread, torch.zeros_like(w.r), zero_edge)
    u0, v0 = (None, None) if pair is None else pair
    lam, u, v = ct.leading_pair(A, p.iterations, u0, v0)
    node_rel, edge_rel = ct.drops_from_pair(A, lam, u, v)
    return Reach(node, edge, tail, spread, node_rel, edge_rel, (u, v))


def inspect_value(w: World, p: Params, rc: Reach) -> Tensor:
    """Value of one audit of agent i: q * min(r * at_stake, (1 - r) * lambda * centrality). Settling a true positive saves
    what its compromise puts at stake, clearing a false positive saves the utility its restriction would cost; the
    smaller of the two bounds what a verdict is worth (the form of bazt.actions.score_inspect)."""
    send = (w.active & (w.live.any(-1))).to(w.r.dtype)
    return send * p.q_inspect * torch.minimum(w.r * rc.node, (1.0 - w.r) * p.utility_weight * centrality(w))


def price(w: World, p: Params, costs: Mapping[str, float], elig: Tensor, kinds: FrozenSet[core.ActionKind],
          topology: bool, rc: Reach | None = None) -> Dict[core.ActionKind, Priced]:
    """Priced candidates for every kind in `kinds` except INSPECT (see inspect_value). `elig` bool [B, N]: the agent
    is suspicious enough to restrict (calibrated threshold or a positive verdict). `rc` is the contagion model when the
    caller already has it (it keeps the eigenpair to warm-start the next round)."""
    out_: Dict[core.ActionKind, Priced] = {}
    A = transmission(w, p)
    rc = reach(w, A, p, topology) if rc is None else rc
    go = hops(w) > 0
    left = float(rounds_left(w))
    C = centrality(w)
    out = out_degree(w.live, w.r.dtype)
    charged = out_degree(w.adj & w.active[:, :, None] & w.active[:, None, :])    # what the platform counts (see header)
    sends = w.active & (out > 0)
    r, lam, rho = w.r, p.utility_weight, p.rho
    own = r * rc.node                                            # expected reach of the agent itself
    others = (rc.spread[:, None] - own).clamp_min(0.0)           # spread that does not start at the agent
    live_elig = w.live & elig[:, :, None]
    share = C / out.clamp_min(1.0)                               # the utility an edge of the sender stands for

    def edge(kind, gain, cost, recurrence):
        net = gain - lam * (1.0 - r)[..., None] * rho[kind.value] * share[..., None]
        out_[kind] = Priced(net, torch.full(A.shape, cost, dtype=torch.float64, device=A.device),
                            torch.full(A.shape, recurrence, dtype=torch.float64, device=A.device), live_elig & go & (net > 0))

    if K.ISOLATION in kinds:
        net = own + rc.node_rel * others - lam * (1.0 - r) * rho["isolation"] * C
        out_[K.ISOLATION] = Priced(net, unit_cost(costs, K.ISOLATION) * charged * left, torch.ones_like(r),
                                   elig & sends & go & (net > 0))
    if K.PRIVILEGE_DEGRADATION in kinds:
        net = p.sanitize_success * own - lam * (1.0 - r) * rho["privilege_degradation"] * C
        out_[K.PRIVILEGE_DEGRADATION] = Priced(net, unit_cost(costs, K.PRIVILEGE_DEGRADATION) * charged * left,
                                               torch.ones_like(r), elig & sends & ~w.degraded & go & (net > 0))
    if K.RATE_LIMIT in kinds:
        eff = (1.0 - p.rate_limit / out.clamp_min(1.0)).clamp_min(0.0)
        net = eff * (own - lam * (1.0 - r) * rho["rate_limit"] * C)
        out_[K.RATE_LIMIT] = Priced(net, unit_cost(costs, K.RATE_LIMIT) * (charged - p.rate_limit).clamp_min(0.0) * left,
                                    torch.ones_like(r), elig & sends & ~w.limited & (eff > 0) & go & (net > 0))
    if K.EDGE_QUARANTINE in kinds:
        edge(K.EDGE_QUARANTINE, r[..., None] * rc.edge + rc.edge_rel * others[..., None],
             unit_cost(costs, K.EDGE_QUARANTINE) * left, 1.0)
    if K.SANITIZE in kinds:
        edge(K.SANITIZE, p.sanitize_success * r[..., None] * rc.edge / max(hops(w), 1), unit_cost(costs, K.SANITIZE), left)
    if K.DELAY in kinds:
        edge(K.DELAY, r[..., None] * rc.tail, unit_cost(costs, K.DELAY), left)
    if K.REINSTATEMENT in kinds:
        out_[K.REINSTATEMENT] = _reinstatement(w, p, costs, topology, C, go)
    return out_


def _reinstatement(w: World, p: Params, costs: Mapping[str, float], topology: bool, C: Tensor, go: bool) -> Priced:
    """Lifting the restrictions of agent i regains the utility they cost a benign agent and re-exposes the swarm to i
    if it is compromised: its reach on the graph where it sends again. Worth doing exactly when the restriction no
    longer pays (the two nets are complements), so no eligibility test is needed and a restriction does not flap.

    The graph where it sends again is `full` (every other quarantine stays) plus its own quarantined out-edges, which
    the lift restores: its reach is H_i + sum_j q_ij R_j (first order in the lifted edges; without them an agent whose
    out-edges are all quarantined would have no reach and be reinstated for free). The lift also reopens the
    quarantined deliveries into i, each worth what its sender may carry: d R_i r_k per edge k -> i, the complement of
    what quarantining it removed (one unit per edge without topology)."""
    full = w.adj & ~w.quarantined
    q = (w.adj & w.quarantined).to(w.r.dtype)
    senders_risk = torch.bmm(q.transpose(1, 2), w.r[..., None])[..., 0]        # sum_k q_ki r_k
    n_full = out_degree(full, w.r.dtype)
    eff_rate = (1.0 - p.rate_limit / n_full.clamp_min(1.0)).clamp_min(0.0)
    touching = (w.quarantined.sum(2) + w.quarantined.sum(1)).to(w.r.dtype)
    q_share = touching / (out_degree(w.adj, w.r.dtype) + w.adj.sum(1).to(w.r.dtype)).clamp_min(1.0)   # edges lifted with it
    q_out = out_degree(w.adj & w.quarantined, w.r.dtype) / out_degree(w.adj, w.r.dtype).clamp_min(1.0)               # its own cut reach
    one = torch.ones_like(w.r)
    zero = torch.zeros_like(w.r)
    eff = torch.stack([torch.where(w.isolated, one, zero),
                       torch.where(w.degraded, one * p.sanitize_success, zero),
                       torch.where(w.limited, eff_rate, zero),
                       torch.where(q_out > 0, q_out, zero)]).amax(0)
    rho = torch.stack([torch.where(w.isolated, one * p.rho["isolation"], zero),
                       torch.where(w.degraded, one * p.rho["privilege_degradation"], zero),
                       torch.where(w.limited, eff_rate * p.rho["rate_limit"], zero),
                       torch.where(q_share > 0, q_share * p.rho["edge_quarantine"], zero)]).amax(0)
    structural = zero
    if topology:
        d = decay(w, p)
        A_full = full.to(w.r.dtype)
        H, R, _ = ct.katz(A_full, _susceptibility(w, p, everyone=True), d, hops(w))
        node_full = d[:, None] * (H + torch.bmm(q, R[..., None])[..., 0])
        reopened = d[:, None] * R * senders_risk
        if bool(w.isolated.any()) or bool(w.quarantined.any()):      # D118: only isolation and quarantine carry structure
            structural = _restored_structure(w, p, A_full, q, d[:, None] * H)
    else:
        on = float(hops(w) > 0)
        node_full = torch.full_like(w.r, on)
        reopened = on * senders_risk
    net = p.utility_weight * (1.0 - w.r) * rho * C - eff * w.r * node_full - reopened - structural
    restricted = w.isolated | w.degraded | w.limited | (q_share > 0)
    # charged per delivery the agent sends again over the remaining rounds (an upper bound: the base graph's out-degree)
    cost = unit_cost(costs, K.REINSTATEMENT) * out_degree(w.adj) * float(rounds_left(w))
    return Priced(net, cost, torch.ones_like(w.r), restricted & go & (net > 0))


def _restored_structure(w: World, p: Params, A_full: Tensor, q: Tensor, node: Tensor) -> Tensor:
    """[B, N] the structural share a reinstatement gives up (D118): what its restrictions were bought for beyond the
    agent's own reach. One extra power iteration (`p.iterations`, cold start, so nothing to checkpoint) on the restored
    graph `A_full` = adj & ~quarantined, where every isolated agent sends again:
      isolated agent i     node_rel_i x others_i                   (NetShield, as in the isolation's price)
      lifted edge k -> j   v_k u_j / (lam v.u) x others_k          (NetMelt to first order, the edge being absent from
                                                                     A_full; as in the quarantine's price), summed over the
                                                                     quarantined edges out of and into the agent
    with others_k the spread that does not start at k in the same restored world: every agent, the isolated ones included,
    on the same graph (`node` = d H, whose Katz reach already counts every agent's edges and susceptibility). With one
    restricted agent this is the spread its isolation was priced against, so isolating agent i in the same world is worth
    exactly minus reinstating it. With several isolated at once, each lift is priced as if the others were restored too
    (the conservative reading: their spread still counts as a threat). An episode with neither an isolated agent nor a
    quarantined edge gets 0, so the result does not depend on the other episodes of the batch."""
    lam, u, v = ct.leading_pair(A_full, p.iterations)
    node_rel, lifted_rel = ct.drops_from_pair(q, lam, u, v)
    own = w.r * node
    others = (own.sum(-1, keepdim=True) - own).clamp_min(0.0)
    return (w.isolated.to(w.r.dtype) * node_rel * others + others * lifted_rel.sum(-1)
            + (lifted_rel * others[:, :, None]).sum(1))

