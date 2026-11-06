"""World construction for a batch of episodes: everything random that must be identical across arms.

A world depends on the unit's (topology, N, seed) keys only: not on the arm, the attack family, the budget cell or the
sink, and not on the other units of the batch (CRN). Streams are keyed by strings (core.stream) and pre-drawn for all
rounds, so a defender can never desynchronise them. Units with the same seed (every agent takes the sink role in turn)
share one draw.

Streams per (topology, N, seed):  "g" the graph, "w0" readings and placements (independent of the horizon), "wT" the
per-round noise, "rate" one lazy draw per (episode, round) for rate-limit priorities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import auditor, graphs
from bazt_lab.features.mock_engine.private.model import Model, load_model

ATTACKS = ("none", "insider_pi", "insider_ta", "insider_ma", "infection", "aitm")
INSIDERS = ("insider_pi", "insider_ta", "insider_ma")
# What a unit's `attack_params` may carry. The first four are the engine's own overrides, the rest is the adaptive
# feature's strategy space (adaptive/private/space.py).
PARAM_KEYS = ("n_insiders", "insider_bias", "mu_inf", "auditor_tpr",
              "entry_ranks", "start_frac", "rate", "detectability", "potency")
# The keys an attack-free run reads: the detection rate of the adaptive auditor preset (part of the defence, set by a
# robustness variant). Every other key describes the attack and changes nothing in a run without one (no agent carries
# the payload), so the attack-free twin of a unit (D109) keeps only these.
ATTACK_FREE_PARAMS = ("auditor_tpr",)
SAME = ("arm", "audit", "intervention", "topology", "attack", "agents", "cost_condition", "preset")
STAR = ("star_hub", "star_leaf")


@dataclass
class World:
    B: int
    N: int
    T: int
    topology: str
    family: str
    seeds: tuple                    # unit seeds, for the lazy keyed draws
    adj: torch.Tensor               # bool [B, N, N] base graph (symmetric)
    deg: torch.Tensor               # int64 [B, N]
    x: torch.Tensor                 # float64 [B, N] private readings
    mean: torch.Tensor              # float64 [B] true mean of the readings
    seed: torch.Tensor              # bool [B, N] agents that carry the payload from the start (entry agent, insiders)
    attackers: torch.Tensor         # bool [B, N] insiders
    victim: torch.Tensor            # int64 [B] AiTM victim, -1 otherwise
    decoy: torch.Tensor             # bool [B, N]
    burst: torch.Tensor             # bool [B, T, N] decoy header burst
    sink: torch.Tensor              # int64 [B]
    shift: torch.Tensor             # float64 [B] reported-value shift of a payload-carrying agent
    mu_mal: torch.Tensor            # float64 [B] mean monitor score of a payload-carrying message
    rewrite: torch.Tensor           # float64 [B] probability an AiTM delivery into the victim is rewritten
    d_pay: torch.Tensor             # float64 [B] auditor mean score of a payload text (benign 0)
    d_dec: torch.Tensor             # float64 [B] auditor mean score of a decoy-header text
    attack_on: torch.Tensor         # bool [B, T] rounds in which the attack acts
    mon_z: torch.Tensor             # float64 [B, T, N] monitor noise (standard normal)
    aud_z: torch.Tensor             # float64 [B, T, N] auditor score noise (standard normal)
    u_adopt: torch.Tensor           # float64 [B, T, N] uniforms
    u_stay: torch.Tensor
    u_decoy: torch.Tensor
    u_strip: torch.Tensor
    u_aitm: torch.Tensor            # [b, t, j]: the delivery from j into the victim is rewritten iff u_aitm < rewrite
    keys: torch.Tensor              # int64 [B]: one key per episode, (topology, N, seed), for the keyed draws below

    FLOATS = ("x", "mean", "shift", "mu_mal", "rewrite", "d_pay", "d_dec", "mon_z", "aud_z", "u_adopt", "u_stay", "u_decoy",
              "u_strip", "u_aitm")

    def on(self, device: torch.device, dtype: torch.dtype) -> World:
        """The same world with every tensor on `device` and the real-valued ones in `dtype` (the draws are made once, in
        float64 on the CPU, then converted: the numbers are the same whatever the device)."""
        moved = {}
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                moved[name] = value.to(device, dtype) if name in self.FLOATS else value.to(device)
            else:
                moved[name] = value
        return World(**moved)


def check_homogeneous(units: Sequence[Mapping]) -> None:
    """A batch shares its arm, budget cell, topology, attack, N, cost condition and auditor preset."""
    if not units:
        raise ValueError("empty batch: no units to run")
    for field in SAME:
        values = {repr(u.get(field)) for u in units}
        if len(values) > 1:
            raise ValueError(f"units of one batch must have the same {field}; got {sorted(values)}")


def metropolis(adj: torch.Tensor) -> torch.Tensor:
    """float64 [B, N, N]: w_ij = 1 / (1 + max(deg_i, deg_j)) on the edges of the base graph (swarmlab)."""
    deg = adj.sum(2).to(torch.float64)
    return adj.to(torch.float64) / (1.0 + torch.maximum(deg[:, :, None], deg[:, None, :]))


# ----------------------------------------------------------------------------------------------------- draws
def _per_seed(seeds: Sequence[int], make) -> Dict[str, torch.Tensor]:
    """make(seed) -> {name: tensor}, evaluated once per distinct seed and stacked back to one row per unit."""
    unique = list(dict.fromkeys(seeds))
    where = {s: k for k, s in enumerate(unique)}
    index = torch.tensor([where[s] for s in seeds], dtype=torch.int64)
    drawn = [make(s) for s in unique]
    return {name: torch.stack([d[name] for d in drawn])[index] for name in drawn[0]}


def make_graphs(units: Sequence[Mapping], settings: Mapping, topology: str, N: int) -> torch.Tensor:
    """bool [B, N, N]: the base graph of each unit, from the stream keyed (g, topology, N, seed)."""
    ws_k = core.get(settings, "runtime.ws_k") if topology == "small_world" else None
    ws_p = core.get(settings, "runtime.ws_rewire_p") if topology == "small_world" else None
    ba_m = core.get(settings, "runtime.ba_m") if topology == "barabasi_albert" else None
    shape = graphs.draw_shape(topology, N, ws_k, ba_m)
    B = len(units)
    if shape is None:
        return graphs.build_graphs(topology, N, None, 1).expand(B, N, N).clone()
    seeds = [u["seed"] for u in units]
    unique = list(dict.fromkeys(seeds))
    u = core.batch_uniforms([("g", topology, N, s) for s in unique], shape)
    adj = graphs.build_graphs(topology, N, u, len(unique), ws_k=ws_k, ws_p=ws_p, ba_m=ba_m)
    where = {s: k for k, s in enumerate(unique)}
    return adj[torch.tensor([where[s] for s in seeds], dtype=torch.int64)]


def _draw(topology: str, N: int, T: int):
    def make(seed: int) -> Dict[str, torch.Tensor]:
        g0, gT = core.stream("w0", topology, N, seed), core.stream("wT", topology, N, seed)
        z0 = torch.randn((N,), generator=g0, dtype=torch.float64)
        u0 = core.uniforms(g0, (5, N))
        zT = torch.randn((2, T, N), generator=gT, dtype=torch.float64)
        uT = core.uniforms(gT, (5, T, N))
        return dict(z0=z0, place1=u0[0], place2=u0[1], decoy_u=u0[2], burst_u=u0[3], sink_u=u0[4, 0],
                    mon_z=zT[0], aud_z=zT[1], u_adopt=uT[0], u_stay=uT[1], u_decoy=uT[2], u_strip=uT[3], u_aitm=uT[4])
    return make


def draw_noise(units: Sequence[Mapping], topology: str, N: int, T: int) -> Dict[str, torch.Tensor]:
    """The arm-independent draws of each unit (readings, placements, per-round noise) for horizon T. They do not depend on
    the attack, so a paired attack-free run of the same episodes can reuse them (`build_world(..., draws=...)`)."""
    return _per_seed([u["seed"] for u in units], _draw(topology, N, T))


def rate_priority(world: World, rows: torch.Tensor, t: int) -> torch.Tensor:
    """float64 [len(rows), N, N]: uniforms that rank the receivers of a rate-limited sender in round t. Keyed by
    (topology, N, seed, t) without the arm (core.keyed_uniform: the same numbers on every device and batch)."""
    return core.keyed_uniform(world.keys[rows.to(world.keys.device)], "rate", t, (world.N, world.N), world.keys.device)


def audit_priority(world: World, rows: torch.Tensor, t: int, call: torch.Tensor, stream: str = "audit") -> torch.Tensor:
    """float64 [len(rows), N]: uniforms that rank the requested senders when an episode asks for more audits than it can
    afford (the lowest are granted). Keyed by (topology, N, seed, round, the episode's own call number), never the arm or the batch. `stream` separates the
    plain audit from the audit of isolated agents (D123), each with its own call number."""
    keys = world.keys[rows.to(world.keys.device)]
    return core.keyed_uniform(keys, stream, t * (1 << 20) + call.to(keys.device), (world.N,), keys.device)


# ------------------------------------------------------------------------------------------- attack parameters
def _params(unit: Mapping) -> dict:
    p = unit.get("attack_params") or {}
    if not isinstance(p, Mapping):
        raise ValueError(f"attack_params must be a mapping, got {type(p).__name__}")
    unknown = sorted(set(p) - set(PARAM_KEYS))
    if unknown:
        raise ValueError(f"attack_params: unknown key(s) {unknown}; the mock knows {list(PARAM_KEYS)}")
    for key, lo, hi in (("start_frac", 0.0, 1.0), ("rate", 0.0, 1.0), ("detectability", 0.0, 1.0), ("potency", 0.0, 1.0)):
        if key in p and not (isinstance(p[key], (int, float)) and not isinstance(p[key], bool) and lo <= p[key] <= hi):
            raise ValueError(f"attack_params[{key!r}] must be a number in [{lo}, {hi}], got {p[key]!r}")
    for key in ("insider_bias", "mu_inf"):
        if key in p and (isinstance(p[key], bool) or not isinstance(p[key], (int, float))):
            raise ValueError(f"attack_params[{key!r}] must be a number, got {p[key]!r}")
    if "n_insiders" in p and (isinstance(p["n_insiders"], bool) or not isinstance(p["n_insiders"], int)
                              or p["n_insiders"] < 1):
        raise ValueError(f"attack_params['n_insiders'] must be an integer >= 1, got {p['n_insiders']!r}")
    if "entry_ranks" in p and (not isinstance(p["entry_ranks"], (list, tuple))
                               or not all(isinstance(r, int) and not isinstance(r, bool) for r in p["entry_ranks"])):
        raise ValueError(f"attack_params['entry_ranks'] must be a list of integers, got {p['entry_ranks']!r}")
    return dict(p)


def _runtime(settings: Mapping, key: str, lo: float | None = None, hi: float | None = None) -> float:
    value = core.get(settings, f"runtime.{key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (lo is not None and value < lo) or (
            hi is not None and value > hi):
        raise ValueError(f"runtime.{key}={value!r} must be a number in [{lo}, {hi}]")
    return float(value)


def _entry(topology: str, place1: torch.Tensor, N: int) -> torch.Tensor:
    """int64 [B]: where a single entry (infection seed, primary insider, AiTM victim) sits: the hub of star_hub, a
    uniformly drawn leaf of star_leaf, a uniformly drawn agent elsewhere."""
    u = place1[:, 0]
    if topology == "star_hub":
        return torch.zeros_like(u, dtype=torch.int64)
    if topology == "star_leaf":
        return (1 + (u * (N - 1)).long()).clamp_(max=N - 1)
    return (u * N).long().clamp_(max=N - 1)


def _ranked(deg: torch.Tensor, ranks: Sequence[int], b: int) -> List[int]:
    """Agents with the given degree ranks of episode b (rank 0 = highest degree, ties by agent index)."""
    order = torch.argsort(-deg[b], stable=True)
    return order[list(ranks)].tolist()


def _place(topology: str, family: str, params: List[dict], deg: torch.Tensor, d: Mapping, n_default: int):
    """(seed, attackers, victim) for the family; an `entry_ranks` override replaces the drawn entry agents."""
    B, N = deg.shape
    seed = torch.zeros(B, N, dtype=torch.bool)
    victim = torch.full((B,), -1, dtype=torch.int64)
    rows = torch.arange(B)
    primary = _entry(topology, d["place1"], N)
    if family == "infection":
        seed[rows, primary] = True
    elif family == "aitm":
        victim = primary.clone()
    elif family in INSIDERS:
        n = torch.tensor([p.get("n_insiders", n_default) for p in params])
        if (n >= N).any():
            raise ValueError(f"n_insiders={int(n.max())} leaves no victim among N={N} agents")
        seed[rows, primary] = True
        key = d["place2"].masked_fill(seed, math.inf)
        seed |= key.argsort(1).argsort(1) < (n - 1)[:, None]
    for b, p in enumerate(params):
        if "entry_ranks" not in p or family == "none":
            continue
        ranks, want = p["entry_ranks"], (int(n[b]) if family in INSIDERS else 1)
        if len(ranks) != want or len(set(ranks)) != want or not all(0 <= r < N for r in ranks):
            raise ValueError(f"attack_params['entry_ranks']={ranks} must hold {want} distinct ranks in [0, {N})")
        agents = _ranked(deg, ranks, b)
        if family == "aitm":
            victim[b] = agents[0]
        else:
            seed[b] = False
            seed[b, agents] = True
    attackers = seed.clone() if family in INSIDERS else torch.zeros_like(seed)
    return seed, attackers, victim


def _decoys(topology: str, deg: torch.Tensor, eligible_out: torch.Tensor, decoy_u: torch.Tensor, count: int):
    """Decoys: `count` agents drawn among the low-degree half of the agents (never a star hub) that are not carrying
    the attack. The pool does not depend on the attack, so a decoy is the same agent in an attack-free run unless
    the attacker takes its place."""
    B, N = deg.shape
    keep = torch.ones(N, dtype=torch.bool)
    if topology in STAR:
        keep[0] = False
    n_keep = int(keep.sum())
    size = min(n_keep, max(4, math.ceil(0.5 * n_keep)))
    by_degree = (deg.double() + (~keep).double() * 1e9).argsort(dim=1, stable=True)
    pool = torch.zeros(B, N, dtype=torch.bool).scatter_(1, by_degree[:, :size], True)
    eligible = pool & ~eligible_out
    key = decoy_u.masked_fill(~eligible, math.inf)
    return (key.argsort(1).argsort(1) < count) & eligible


def _burst(burst_u: torch.Tensor, T: int, span: int) -> torch.Tensor:
    """bool [B, T, N]: the decoy header is written in a burst of 1 or 2 rounds starting at round 1 + (i % span)."""
    N = burst_u.shape[1]
    span = max(1, min(span, T))
    start = 1 + torch.arange(N) % span
    length = 1 + (burst_u < 0.5).long()
    t = torch.arange(1, T + 1)[None, :, None]
    return (t >= start[None, None, :]) & (t <= start[None, None, :] + length[:, None, :] - 1)


def _schedule(params: List[dict], T: int) -> torch.Tensor:
    on = torch.ones(len(params), T, dtype=torch.bool)
    for b, p in enumerate(params):
        if "start_frac" in p or "rate" in p:
            from bazt_lab.features.adaptive import public as adaptive
            on[b] = adaptive.attack_rounds(T, p.get("start_frac", 0.0), p.get("rate", 1.0))
    return on


def _family_values(family: str, settings: Mapping, model: Model, params: List[dict]):
    """(shift, mu_mal, rewrite, n_insiders default) of each episode: swarmlab's bias and mu_inf for the infection,
    the run-time insider variables for the insider families, the AiTM rewrite strength for AiTM."""
    B = len(params)
    shift, mu, rewrite, n_default = 0.0, model.mu_inf, 0.0, 0
    if family in INSIDERS:
        n_default = int(_runtime(settings, "n_insiders", 1))
        shift = _runtime(settings, "insider_bias")
        mu = {"insider_pi": model.mu_inf, "insider_ta": _runtime(settings, "ta_detectability"),
              "insider_ma": _runtime(settings, "ma_detectability")}[family]
    elif family in ("infection", "aitm"):
        shift = model.bias
        rewrite = _runtime(settings, "aitm_rewrite_strength", 0.0, 1.0) if family == "aitm" else 0.0
    one = lambda v: torch.full((B,), float(v), dtype=torch.float64)       # noqa: E731
    shift, mu, rewrite = one(shift), one(mu), one(rewrite)
    for b, p in enumerate(params):
        if family in INSIDERS and "insider_bias" in p:
            shift[b] = p["insider_bias"]
        if "mu_inf" in p:
            mu[b] = p["mu_inf"]
    if family == "none":
        shift = torch.zeros_like(shift)
    return shift, mu, rewrite, n_default


def build_world(units: Sequence[Mapping], settings: Mapping, topology: str, N: int, family: str, adj: torch.Tensor,
                T: int, draws: Optional[Mapping[str, torch.Tensor]] = None) -> World:
    if family not in ATTACKS:
        raise ValueError(f"unknown attack {family!r}; the mock knows {ATTACKS}")
    model = load_model(settings)
    B = len(units)
    params = [_params(u) for u in units]
    seeds = tuple(u["seed"] for u in units)
    d = draws if draws is not None else draw_noise(units, topology, N, T)
    deg = adj.sum(2)
    x = model.reading_mean + model.reading_sd * d["z0"]
    shift, mu_mal, rewrite, n_default = _family_values(family, settings, model, params)
    seed, attackers, victim = _place(topology, family, params, deg, d, n_default)
    decoy = _decoys(topology, deg, seed | attackers, d["decoy_u"], model.decoys)
    sink = torch.tensor([-1 if u.get("sink") is None else u["sink"] for u in units], dtype=torch.int64)
    if ((sink >= N) | (sink < -1)).any():
        raise ValueError(f"sink must be an agent index in [0, {N}) or None")
    drawn = (d["sink_u"] * N).long().clamp_(max=N - 1)
    sink = torch.where(sink >= 0, sink, drawn)
    models: Dict[tuple, auditor.AuditorModel] = {}
    d_pay, d_dec = torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)
    for b, (u, p) in enumerate(zip(units, params, strict=True)):
        key = (u.get("preset"), p.get("auditor_tpr"))
        if key not in models:
            models[key] = auditor.auditor_model(settings, key[0], key[1])
        d_pay[b], d_dec[b] = models[key].d_payload, models[key].d_decoy
    for b, p in enumerate(params):
        if "detectability" in p:
            mu_mal[b] *= p["detectability"]
            d_pay[b] *= p["detectability"]
        if "potency" in p:
            shift[b] *= p["potency"]
    return World(
        B=B, N=N, T=T, topology=topology, family=family, seeds=seeds, adj=adj, deg=deg, x=x, mean=x.mean(1), seed=seed,
        attackers=attackers, victim=victim, decoy=decoy, burst=_burst(d["burst_u"], T, model.burst_span), sink=sink,
        shift=shift, mu_mal=mu_mal, rewrite=rewrite, d_pay=d_pay, d_dec=d_dec, attack_on=_schedule(params, T),
        mon_z=d["mon_z"], aud_z=d["aud_z"], u_adopt=d["u_adopt"], u_stay=d["u_stay"], u_decoy=d["u_decoy"],
        u_strip=d["u_strip"], u_aitm=d["u_aitm"],
        keys=core.episode_keys([f"{topology}|{N}|{s}" for s in seeds]))
