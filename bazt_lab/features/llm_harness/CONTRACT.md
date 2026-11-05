# llm_harness: contract

`LLMPlatform`, the `core.Platform` of the LLM side (`EXPERIMENT_PLAN.md` §3, §4.1, §5), over the G-Safeguard submodule
`vendor/G-safeguard` (pinned at `890c99f1cbc864e9ff0c85859619a14f42bc9cab`, no licence, never copied into git).
Our changes to it are the patch files in `patches/`, applied to a derived copy.

## public.py

| name | what |
|---|---|
| `LLMPlatform(backend=None, judge=None, sample=None, families=None)` | the platform. Constructing it does nothing; the arguments replace what settings select (tests) |
| `FakeBackend`, `fake_sample` | the offline LLM (canned agents, insiders included) and synthetic mmlu/csqa/gsm8k items |
| `Backend`, `Completion` | the seam: `async complete(messages, model) -> Completion(text, prompt_tokens, completion_tokens)` |
| `TOPOLOGIES`, `ATTACKS` | supported topologies (`random_directed`, `hub_star`, `barabasi_albert`, `watts_strogatz`, `chain`, `tree`, `wheel`) and attacks (`none`, `insider_pi/ta/ma`, `infection`, `aitm`) |

`run_batch(units, arm, settings, store, tracker)`: per unit one seed run; a done unit is skipped (its saved rows are
returned). `calibrate(units, arm, settings, store)`: attack-free, uncapped units; returns `{statistic: threshold}`.
`llm.api = "fake"` selects FakeBackend + FakeJudge + synthetic items (no network, no key, no kickoff guard);
anything else is the live path.

## The patch set (`patches/`, applied by `adapters/upstream.prepare`)

`prepare` checks the submodule `HEAD` is the pinned commit, copies the upstream `PI TA MA scalability` `.py` files (LF)
to `results/experiments/_g-safeguard-patched`, runs `git apply --check` then `git apply` for each patch in order, and
stamps the copy with commit + patch digest (a matching stamp is reused). The submodule tree is never modified.

1. `0001-seeds`: `--seed` (required) on the upstream scripts that draw graphs, attackers, samples, wrong answers.
2. `0002-imports-and-paths`: TA/MA/scalability import errors (nltk, sentence_transformers, `gen_graph` import, the
   `prompts`/`prompt` typo in scalability), `tools.json` path, gsm8k import.
3. `0003-defender-hook`: the GNN block of `defense_communication` is a call `hook(turn, responses)`.
4. `0004-delivery-masks`: `AgentGraphWithDefense.delivery` (a plan with `.view(src, dst)`); only delivered edges reach a prompt.
5. `0005-sender-outbox`: `Agent.outbox`, the sender's record of what it sent; a denied, sanitized or delayed delivery
   never touches the sender's chat memory (D10).
6. `0006-external-context`: `AgentGraphWithDefense.context`, a document appended to one agent's question (infection).

Replacing the pinned Linux CUDA wheels (plan §4.1 item 3) is `pyproject.toml`/`uv.lock` (quality), not a patch.

## A seed run (`private/seedrun.py`)

One unit = `unit["conversations"]` conversations on one graph (`runtime.random_edge_probability`, `ba_m`, `ws_k`,
`ws_rewire_p`), `unit["agents"]` agents, `unit["rounds"]` rounds (1 initial + `rounds - 1` delivery rounds). Agent
chat memory is new in every conversation. Defender state, budgets and persistent actions (isolation, quarantine, rate
limit, degradation) carry over the seed run; `defender.reset(ctx)` is called once.

Per delivery round the platform builds the `Observation` (`t` is the global delivery round of the seed run, `T = (rounds - 1) x conversations`, and
`round_in_conversation` is the 1-based round inside the conversation: the conversation boundary is `round_in_conversation == 1`;
`texts[0][i]` = what agent i wrote, `monitor` None, `authentic` False on AiTM-tampered deliveries, `active` False only
for isolated agents, `attempts` = live deliveries not quarantined), calls `decide`, applies the `Actions`
(`private/enforce.py`: the mock's order and pricing, including that isolation charges only the deliveries still live after
earlier quarantines and rate limits and that reinstating an isolated agent restores one round fewer; `tests/test_majors.py` compares
it with `mock_engine.public.apply_actions` on 120 random Actions), and passes the result to G-Safeguard as a delivery plan:

- unmodified: the sender's last output; sanitized: neutral redaction (answer `None`, reason "[content removed by the
  security gateway]"); delayed: delivered one round later if no fresh delivery arrives on that edge; denied, quarantined,
  rate-dropped, isolated: not delivered. The sender's own memory is untouched in all cases.
- `obs.audit(request [1,N] bool)`: judge score (`Judge.score`) of the requested senders' texts, NaN where not requested or
  unaffordable; one audit unit (`costs.<cond>.inspect`) per text. Over-request is granted to a keyed random subset
  (`stream("llm_harness","audit",topology,N,seed,t)`).
- Budgets are fractions of all deliveries of the seed run: `edges x (rounds - 1) x conversations`. Persistent actions are
  priced over the rounds remaining in the whole seed run. Over-budget actions of a round are all rejected (violation).
- `ctx`: `B=1`, `T = (rounds - 1) x conversations` (the seed-run clock, D54; budgets, persistent-action prices and
  `enforce` use the same `t` and `T`), `platform="llm"`, `cell={"attack", "task"}` (the GNN arms pick their family from it with
  `gnn.deploy_family`; the harness does not choose a family), `sink` = the position of the current conversation (D100: conversation k
  uses `runtime.llm_sink_positions[topology][k mod len]`; `ctx.sink` is updated in place), `thresholds` from the unit.
- Sanitize (D101), `unit["sanitize_mode"]` (required for evaluation units; refused if unknown): `redaction` is the fixed redaction;
  `llm_rewrite` is one call per sanitized text on the agent backbone with the fixed prompt of `private/prompts.py` (strip instructions,
  keep content); the receiver gets the original answer and the rewritten reason. Tokens: `tokens_sanitizer`.

Common random numbers: graph, placement, sink, items, wrong answers and rate-limit draws are functions of string keys
(seed, topology, N, conversation, round) and never of the arm. The harness keeps no generator, so a checkpoint has no
generator state.

### Attacks (D5, D38)

| attack | implementation |
|---|---|
| `insider_pi` | G-Safeguard PI attacker prompts (mmlu, csqa; gsm8k "disruptor"), `runtime.n_insiders` insiders |
| `insider_ta` | G-Safeguard TA (InjecAgent scratchpad), success = `TA/utils.evaluate_output_prompted` |
| `insider_ma` | G-Safeguard MA, attackers read the 5 poisoned passages; correct = normalised containment of the answer text |
| `infection` | `private/prompts.py`: Lee & Tiwari 2024 (arXiv 2410.07283) App. A Fig. 8, in the entry agent's context. Adaptation: the payload step is also in the replicating branch (no "last agent" in a graph) |
| `aitm` | He et al. ACL Findings 2025 (arXiv 2502.14847) §3.3 reflection prompt + Table 7 role text, one adversary call per round, instruction appended to every delivery to the victim; those deliveries have `authentic` False |

Memory-attack correctness (D103): `llm.ma_correctness` (list of `rule`, `llm_judge`); an MA unit gets one row per mode from the
same conversation (`ma_check`; no extra agent calls). `llm_judge` is the upstream judge (system prompt read at run time from
`MA/evaluate_output.py` of the patched copy, user prompt as upstream, "Reply with only 0 or 1." added) on `llm.ma_judge_model`, one
call per agent and round (`tokens_ma_judge`, only in that row). Other units: one row, `ma_check` None.

Placement: one permutation of `stream("llm_harness","placement",topology,N,seed)`; insiders = first `n_insiders`, infection
entry agent and AiTM victim = first. Sink = a draw from `runtime.llm_sink_positions.<topology>`.

### Rows and traces

One `EpisodeTrace` (B=1, T=rounds) per conversation, turned into a row by `metrics.rows_from_trace`; round 0 = the initial
answers. `unit_ids[0]` = `{id, block, platform, arm, audit, intervention, cost_condition, backbone, topology, attack, task,
seed, conversations, agents, rounds, conversation, item}` (None where the unit has none, so rows share their keys); the row
also has `sink`. Isolation losses stay in `attempts`. `meta`: `tampered` (AiTM), `flagged` (rate-limited or degraded),
`tokens` `{agent, adversary, judge}`, `latency`, `seeds`, `events` (defender events such as `budget_exhausted`),
`exhausted_global_round`. `infected` = reply carries the payload or the agent is an insider; `correct` = right answer
(tool attack: attacker tool not called); `goal_reached` = a victim carries the payload in the last round.
Every unit field reaches the row (`id, block, platform, arm, audit, intervention, cost_condition, backbone, topology, attack, task,
seed, conversations, agents, rounds, sanitize_mode, condition, preset` are present as None when absent); `meta_providers` and `meta_quantizations` (sorted lists of what the responses reported; the live backend keeps them beside the
cached text so cache hits report them too; FakeBackend reports provider `fake`), `meta_judge_off_format` and
`meta_judge_missing_yes` are the judge's format counters over the conversation. `exhausted_step` is the 1-based round within the conversation (1 if already exhausted at its start).

Checkpoint: after every conversation `store.append_rows` (unless the row is already there), then
`store.save_state(unit id, {conv, defender.state_dict(), budgets, persistent state, exhausted, calibration values})`.
A crash between the two re-runs that conversation (deterministic, cached) without a duplicate row.

### Attack-free twin

Graph, placement, items, wrong answers and sink sequence are functions of (seed, topology, task, N, conversation index) only, so a
unit and its attack-free twin (same arm, budget cell, backbone, topology, task, seed) see the same graph, questions and sinks per
conversation index (tested). Rows carry `conversation`; analysis pairs on it.

### Calibration

`calibration.mode` (owner key, no default; unset refuses naming it): `open_loop` = one pass with the arm acting on nothing;
`closed_loop` = thresholds are the `(1 - a)` quantiles of the open-loop statistics and `a` is searched (start at alpha, double while
the realised deny rate is below target, then bisection, at most `CLOSED_LOOP_PASSES` = 10 passes, deterministic, every pass saved
under `<unit id>.calib.p<k>` so a repeat reads them back) until the realised restricted share of open deliveries is alpha within
`CLOSED_LOOP_TOLERANCE` = 0.25 x alpha. `platform.last_calibration` = `{mode, passes, rate, alpha, converged}`; an arm whose
rate cannot reach the tolerance returns the closest pass with `converged` False.

Each pass runs the unit's conversations attack-free and uncapped, collects `Actions.statistics` (`[1,N,N]` over open deliveries,
`[1,N]` over active agents; non-finite entries dropped), and returns the `1 - budgets.alpha` quantile (`higher`) per
statistic over all units. State is saved under `<unit id>.calib`, so a repeated call reads it back.

### gnn_train units (`edge_probability` in the unit)

Attacked conversations without any defence on a `random_directed` graph at that edge probability with `attackers`
insiders; each conversation is logged with `gnn.log_conversations`. The store gets one progress row per conversation (`id, block, arm,
attack, task, conversation, ma_check None, item, record, distinct_questions`), not metrics rows. Upstream training graphs (spec 2.7): the unit is one
(edge probability, attackers) setting, its conversations are spread over `gnn_train.graphs_per_setting` random graphs, each with its
own attacker set (keyed draws). Training questions never include a question that an evaluation or calibration unit of the task
draws (D104): the excluded set is computed from `runtime.llm_seeds`, `runtime.llm_calibration_seeds`, `llm.conversations_per_cell`
and `llm.calibration_conversations_per_cell`, and the unit draws `k + |excluded|` (or the whole pool when it has fewer) and keeps the first `k` others. When fewer than `k`
questions remain, the conversations draw from them keyed WITH repetition, as upstream's graph x question pairs do; the unit refuses only
when none remains. The number of distinct questions is `distinct_questions` in every training record (with `question`).
On every `run_batch` containing gnn_train units, for each such family with `gnn.training_finished(settings, family)` False,
`gnn.train_unit` is called again (a `TrainingDataIncomplete` is ignored), so a crash during training resumes on relaunch.

## Settings keys read

`runtime.n_insiders` (insider attacks), `runtime.rate_limit_per_identity`, `runtime.random_edge_probability`,
`runtime.ba_m`, `runtime.ws_k`, `runtime.ws_rewire_p` (the topology's), `runtime.llm_sink_positions.<topology>`,
`runtime.recovery_margin` (metrics), `runtime.llm_seeds`, `runtime.llm_calibration_seeds`, `llm.conversations_per_cell`,
`llm.calibration_conversations_per_cell` (gnn_train questions), `calibration.mode`, `llm.ma_correctness`, `llm.ma_judge_model`,
`gnn_train.graphs_per_setting` (gnn_train units), `costs.<main|robustness>.<every ActionKind>`, `budgets.alpha` (calibrate),
`llm.api` (optional; `"fake"`), and on the live path `llm.allow_live`, `llm.providers` (backbone -> provider slug),
`llm.temperature`, the judge keys of `judge` (`llm.judge.*`, `llm.cache`), plus `llm_data.*` (sampler) and `gnn_train.*`
(through gnn). A missing key raises `core.Unset` naming it. Secrets: `OPENROUTER_API_KEY` from the environment only.

## Tests

```
.venv\Scripts\python.exe -m pytest bazt_lab/features/llm_harness -q
```
Offline: graphs, enforcement, prompts, cases, the patch set on the pinned checkout, the patched modules, and the
platform end to end with FakeBackend and FakeJudge (2-conversation seed runs for PI, TA, MA, infection, AiTM; crash and
resume; calibration; budgets; gnn_train).

## Open questions

1. (closed by D100) the sink rotates.
2. (closed) `obs.t`, `obs.T` and `ctx.T` count the whole seed run; `round_in_conversation` carries the conversation boundary.
3. (closed by D101) both sanitize modes exist.
4. AiTM goal: wrong-answer forcing (not the paper's encryption-transform goal); adversary model = the agent backbone.
5. (closed by D103) both correctness modes exist.
6. Attack-free TA units still contain InjecAgent's injection in the scratchpad (the item is inherent).
7. Infection adaptation (payload in both branches) is ours.

## Requests

0. gnn_train graphs: add `gnn_train.graphs_per_setting` (upstream: 20) to profiles.toml; the owner value is not set here.
0. `calibration.mode`, `llm.sanitize_modes`, `llm.ma_correctness`, `llm.ma_judge_model` are owner keys, no value chosen here.
1. runner: pass the `--allow-live-llm` decision as `llm.allow_live` in the settings (live path reads it; unset refuses).
2. profiles.toml: `llm.providers` (backbone slug -> pinned provider), `llm.api` is optional (`"fake"` only for tests/e2e).
3. runner/metrics: `clean_accuracy` pairing is not available inside a platform (the attack-free run is another unit);
   analysis should pair it. `meta["clean_accuracy"]` is not set.
4. gnn: `train_unit` loads the real MiniLM when called with `embed=None`; the e2e fake run needs it offline.

## Known gaps

- No live call was made: `adapters/live.py` (LiveBackend, cache lookup of tokens, provider pin) is untested beyond import.
- TA and MA run end to end only with FakeBackend; real-model behaviour is unchecked.
- `clean_accuracy` is not filled (request 3); `recovery_time_accuracy` is therefore NaN.
- GNN training questions: for the vendored tasks (injecagent 771, poisonedrag 100 records) the pool left after the evaluation and
  calibration draws may be smaller than the training conversations asked; the unit then refuses with the counts.
- Closed-loop calibration costs up to 10 extra passes of the calibration conversations (cached requests are paid once).
- Delayed deliveries that are sanitized or rewritten are not re-sanitized when they arrive late.
- Replacing the Linux CUDA wheels is not a patch here (quality owns `pyproject.toml`).
- prompts tests were written together with the module, not before it.
